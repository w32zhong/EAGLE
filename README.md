```sh
cd eagle/traineagle3
DS_SKIP_CUDA_CHECK=1 deepspeed --num_gpus 4 main.py --deepspeed_config ds_config.json
```
