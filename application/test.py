import os
import sys
sys.path.insert(0, '.')
import time
import torch
from model.ea_model import EaModel
import transformers
print(transformers.__path__)

model = EaModel.from_pretrained(
    base_model_path='meta-llama/Llama-2-7b-chat-hf',
    ea_model_path='yuhuili/EAGLE-llama2-chat-7B' if False else './convert_ckpt', # convert_ckpt uses orignal config.json and model.safetensors trained via our pipeline
    torch_dtype=torch.bfloat16,
    #load_in_8bit=True,
    device_map="auto"
)
model.eval()

prompt = "[INST] Thomas is very healthy, but he has to go to the hospital every day. What could be the reasons? [/INST]"
input_ids = model.tokenizer([prompt], return_tensors="pt").input_ids
input_ids = input_ids.to('cuda:0')
print(model.tokenizer.batch_decode(input_ids))
past_len = input_ids.shape[1]
start_time = time.time()
cnt_tokens = 0
for output_ids in model.ea_generate(input_ids, max_length=512):
    #os.system('clear')
    decode_ids = output_ids[0, past_len:].tolist()
    cnt_tokens += len(decode_ids)
    past_len = output_ids.shape[1]
    text = model.tokenizer.decode(decode_ids)
    print(text, end=' ', flush=True)
print()

time_delta = time.time() - start_time
print('e2e speed:', time_delta, cnt_tokens, cnt_tokens / time_delta)
