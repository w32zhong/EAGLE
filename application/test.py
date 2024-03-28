import os
import sys
sys.path.insert(0, '.')
import time
import torch
from model.ea_model import EaModel

model = EaModel.from_pretrained(
    base_model_path='NousResearch/Llama-2-7b-chat-hf',
    ea_model_path='yuhuili/EAGLE-llama2-chat-7B',
    torch_dtype=torch.float16,
    load_in_8bit=True,
    device_map="auto"
)
model.eval()

prompt = '[INST] tell me a few interesting facts about the sun and the moon. [/INST]'
input_ids = model.tokenizer([prompt], return_tensors="pt").input_ids
past_len = input_ids.shape[1]
print(prompt)
start_time = time.time()
for output_ids in model.ea_generate(input_ids, max_steps=512):
    #os.system('clear')
    decode_ids = output_ids[0, past_len:].tolist()
    past_len = output_ids.shape[1]
    text = model.tokenizer.decode(decode_ids)
    print(text, end=' ', flush=True)
print()

time_delta = time.time() - start_time
out_tokens = len(decode_ids)
print('e2e speed:', time_delta, out_tokens, out_tokens / time_delta)
