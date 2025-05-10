# wandb login
# CUDA_VISIBLE_DEVICES=0,1,2,3

#mkdir -p ./ckpt_llama3
#accelerate launch -m --mixed_precision=bf16 eagle.train.main \
#    --tmpdir /workspace/beagle/output/llama3_chat/sharegpt_0_67999_mufp16/ \
#    --cpdir ./ckpt_llama3 --configpath ./eagle/train/EAGLE-LLaMA3-Instruct-8B \
#    --basepath ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a/ \
#    --gradient-accumulation-steps 8 --bs 2

mkdir -p ./ckpt_vicuna
accelerate launch -m --mixed_precision=bf16 eagle.train.main \
    --tmpdir /workspace/beagle/output/vicuna/sharegpt_0_67999_mufp16/ \
    --cpdir ./ckpt_vicuna --configpath ./eagle/train/vicuna_7B_config.json \
    --basepath ~/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.3/snapshots/236eeeab96f0dc2e463f2bebb7bb49809279c6d6 \
    --gradient-accumulation-steps 8 --bs 2
