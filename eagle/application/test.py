import time
import sys
sys.path.insert(0, '.')

import torch
import transformers
from eagle.model.ea_model import EaModel
print(transformers.__path__)

model = EaModel.from_pretrained(
    base_model_path='meta-llama/Llama-3.1-8B-Instruct',
    ea_model_path='w32zhong/confused-snow-233__pondering_ttt12',
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_eagle3=True,
)

messages = [
    {"role": "system",
     "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."}
]
question = "Thomas is very healthy, but he has to go to the hospital every day. What could be the reasons?"
messages.append({
    "role": "user",
    "content": question
})
prompt = model.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

model.eval()

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
