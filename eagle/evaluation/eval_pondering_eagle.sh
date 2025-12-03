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
  "w32zhong/resilient-paper__annealing100_ep5_step_1465K" \
  ; do

  for tree in \
     6,10,50   6,10,60   6,10,70   6,10,80   6,10,90   6,10,100 \
     7,10,50   7,10,60   7,10,70   7,10,80   7,10,90   7,10,100 \
     8,10,50   8,10,60   8,10,70   8,10,80   8,10,90   8,10,100 \
     9,10,50   9,10,60   9,10,70   9,10,80   9,10,90   9,10,100 \
    10,10,50  10,10,60  10,10,70  10,10,80  10,10,90  10,10,100 \
    11,10,50  11,10,60  11,10,70  11,10,80  11,10,90  11,10,100 \
    12,10,50  12,10,60  12,10,70  12,10,80  12,10,90  12,10,100 \
    ; do

    IFS=',' read -r model train_ttt <<< $model_and_train_ttt
    IFS=',' read -r depth top_k total_k <<< $tree

    #options="stats_verbose_0_avg stats_verbose_0_max stats_verbose_0_min stats_cost_0"
    options="disabled0 greedy0_avg"

    for pondering_options in $options; do
      for pondering_threshold in 0.8 0.9 0.95 0.99 1.0; do

        if [[ "$pondering_options" =~ "disabled" ]]; then
          session=$(experiment_sanitize "${model}_${tree}_baseline_${pondering_options}")
        else
          session=$(experiment_sanitize "${model}_${tree}_${pondering_threshold}_${pondering_options}")
        fi

        # any existing log? if yes, evaluate the speeds and skip.
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
