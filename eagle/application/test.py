import os
import sys
sys.path.insert(0, '.')
import time
import torch
from model.ea_model import EaModel
from fastchat.model import get_conversation_template
import transformers
print(transformers.__path__)

model = EaModel.from_pretrained(
    base_model_path='/mnt/cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/',
    ea_model_path='yuhuili/EAGLE-llama2-chat-7B',
    torch_dtype=torch.bfloat16,
    device_map="auto",
    total_token=-1
)
model.eval()
model.ea_layer.tokenizer = model.tokenizer

sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
question = "Thomas is very healthy, but he has to go to the hospital every day. What could be the reasons?"
conv = get_conversation_template("llama-2-chat")  
#conv.system_message = sys_p
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = model.tokenizer([prompt], return_tensors="pt").input_ids
input_ids = input_ids.to('cuda:0')
print(model.tokenizer.batch_decode(input_ids))
past_len = input_ids.shape[1]
start_time = time.time()
cnt_tokens = 0
accept_length = []
cnt = 0
for output_ids in model.ea_generate(input_ids):
    decode_ids = output_ids[0, past_len:].tolist()
    accept_length.append(len(decode_ids))
    cnt += 1
    past_len = output_ids.shape[-1]
    text = model.tokenizer.decode(decode_ids)
    print(text, end=' ', flush=True)
print()

time_delta = time.time() - start_time
print('e2e speed:', sum(accept_length) / time_delta)
print('max accept_length:', max(accept_length))
print('min accept_length:', min(accept_length))
print('avg accept_length:', round(sum(accept_length) / cnt, 3))
print(accept_length)
