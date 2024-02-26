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

prompt = '[INST] How long did the Hundred Years War last? Give a short answer, keep your answer concise. [/INST]'
print(prompt)
input_ids = model.tokenizer([prompt], return_tensors="pt").input_ids
input_len = input_ids.shape[1]
start_time = time.time()
for output_ids in model.ea_generate(input_ids, max_steps=512):
    decode_ids = output_ids[0, input_len:].tolist()
    text = model.tokenizer.decode(decode_ids)
    print(text)
print()

time_delta = time.time() - start_time
out_tokens = len(decode_ids)
print(time_delta, out_tokens, out_tokens / time_delta)
