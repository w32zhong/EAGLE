An example test command:
```sh
CUDA_VISIBLE_DEVICES=0,1 python -m eagle.evaluation.gen_ea_answer_llama3chat --use_eagle3 --model-id test --max-new-token 2048 --base-model-path meta-llama/Meta-Llama-3.1-8B-Instruct --ea-model-path w32zhong/confused-snow-233__pondering_ttt12 --depth 15 --top-k 1 --total-token 17 --pondering_threshold 0.8 --pondering_options stats --question-end 1
```
