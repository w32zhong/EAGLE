import os
import sys
sys.path.insert(0, '.')
import time
import torch
from model.ea_model import EaModel
import transformers
print(transformers.__path__)

model = EaModel.from_pretrained(
    base_model_path='NousResearch/Llama-2-7b-chat-hf',
    ea_model_path='yuhuili/EAGLE-llama2-chat-7B',
    #ea_model_path='w32zhong/s3d-EAGLE-retrain-20K',
    torch_dtype=torch.float16,
    #load_in_8bit=True,
    load_in_4bit=True,
    device_map="auto"
)
model.eval()

prompt = '[INST] tell me something interesting about the solar eclipse in April 2024. [/INST]'
input_ids = model.tokenizer([prompt], return_tensors="pt").input_ids
input_ids = input_ids.to('cuda:0')
past_len = input_ids.shape[1]
print(prompt)
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
