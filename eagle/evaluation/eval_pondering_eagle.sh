source $(dirname $0)/eval_utils.sh
GPUS=$(experiment_argparse --gpus 1 $@)
GPU0=$(experiment_argparse --gpu0 0 $@)
TP_SIZE=$(experiment_argparse --tp_size 1 $@)
QUESTION_END=$(experiment_argparse --end "" $@)
SESSION_END=$(experiment_argparse --session-end "exit" $@)

mkdir -p ./mt_bench
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

for model in \
  "w32zhong/wandering-energy__PonderEagle_ttt10_ep5_tau5_layer1_datacombined_C1.22_23.17_tf32False" \
  "w32zhong/glowing-jazz__PonderEagle_ttt10_ep5_tau5_layer1_datacombined_C1.40_22.90_tf32False" \
  "w32zhong/golden-valley__PonderEagle_ttt10_ep5_tau5_layer1_datacombined_C1.40_22.90_tf32True" \
  "w32zhong/azure-wood__PonderEagle_ttt12_ep5_tau5_layer1_datacombined_C1.40_22.90_tf32False" \
  "w32zhong/snowy-microwave__PonderEagle_ttt10_ep5_tau5_layer2_datacombined_C1.86_22.94_tf32False" \
  "w32zhong/resilient-paper__annealing100_ep5_step_1465K" \
  ; do

  for tree in \
    6,10,70 6,10,80 \
    12,10,90 12,10,80 \
    ; do
    IFS=',' read -r depth top_k total_k <<< $tree

    #options="stats_cost_1ML_avg"

    if [[ "$model" =~ "resilient-paper" ]]; then
      if [ $depth -eq 6 ]; then
        options="disabled"
      else
        options="greedy0_avg"
      fi
    else
      if [ $depth -eq 6 ]; then
        options="disabled_ML"
      else
        options="greedy_1ML_avg greedy_0ML_avg"
      fi
    fi

    for pondering_options in $options; do
      for pondering_threshold in 0.99 1.0; do

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

        # allocate devices
        devices=$(experiment_alloc_devices $cnt $GPU0 $GPUS $TP_SIZE)
        let 'cnt+=1'

        # run a new experiment
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
