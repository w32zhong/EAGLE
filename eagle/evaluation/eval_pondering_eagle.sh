source $(dirname $0)/eval_utils.sh
GPUS=$(experiment_argparse --gpus 1 $@)
GPU0=$(experiment_argparse --gpu0 0 $@)
TP_SIZE=$(experiment_argparse --tp_size 1 $@)
QUESTION_END=$(experiment_argparse --end "" $@)
SESSION_END=$(experiment_argparse --session-end "exit" $@)

mkdir -p ./mt_bench
rm -f gpu_*.lock
cnt=0

run() {
  devices=$1
  session=$2
  shift 2
  experiment_session $session \
    "(flock 200; CUDA_VISIBLE_DEVICES=$devices \
        python -m eagle.evaluation.gen_ea_answer_llama3chat \
          --use_eagle3 --model-id $session --max-new-token 2048 $@ \
        ; flock --unlock 200) 200>gpu_${devices}.lock"
  experiment_session $session $SESSION_END
}

for model_and_train_ttt in \
  "w32zhong/confused-snow-233__pondering_ttt12,12" \
  "w32zhong/fearless-river-231__pondering_ttt10,10" \
  "w32zhong/neat-hill-232__pondering_ttt8,8" \
  "w32zhong/toasty-durian-227__tau3,5" \
  ; do
  for tree in 5,1,7   8,1,10  10,1,12  12,1,14  15,1,17  20,1,22  \
              5,5,25  8,5,40  10,5,50  12,5,60  15,5,75  20,5,100; do
    IFS=',' read -r model train_ttt <<< $model_and_train_ttt
    IFS=',' read -r depth top_k total_k <<< $tree
    if [ $top_k -eq 1 ]; then
      extra_options="joint greedy"
    else
      extra_options="joint_max joint_min joint_avg greedy_max greedy_min greedy_avg"
    fi
    for pondering_threshold in 0.8; do
      for pondering_options in disabled random $extra_options; do
        # (optional) skip extrapolation
        #if [ $depth -gt $train_ttt ]; then continue; fi

        # any existing log? if yes, evaluate the speeds and skip.
        session=$(experiment_sanitize "${model}_${tree}_${pondering_threshold}_${pondering_options}")
        if [ -e mt_bench/$session-*.jsonl ]; then
          python eagle/evaluation/eval_speed.py mt_bench/$session-*.jsonl
          continue
        fi

        # any existing experiment session? if yes, skip.
        if tmux has-session -t "exp_$session"; then
          echo "session exists: exp_$session"; continue
        fi

        # allocate devices and run a new experiment
        devices=$(experiment_alloc_devices $cnt $GPU0 $GPUS $TP_SIZE)
        let 'cnt+=1'
        run $devices $session \
          --base-model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
          --ea-model-path $model \
          --depth $depth --top-k $top_k --total-token $total_k \
          --pondering_threshold $pondering_threshold \
          --pondering_options $pondering_options
      done
    done
  done
done

echo "Total experiments to run: $cnt"
