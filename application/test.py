import os
import sys
sys.path.insert(0, '.')
import time
import torch
from model.ea_model import EaModel
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
print(transformers.__path__)

prompt = '[INST] tell me something interesting about the solar eclipse in April 2024. [/INST]'
max_new_tokens = 256


def test_eagle(bits, random_top_layer=False, quantize_top_layer=False):
    if bits == 4:
        kwargs = dict(load_in_4bit=True, device_map='cuda:0')
    elif bits == 8:
        kwargs = dict(load_in_8bit=True, device_map='cuda:0')
    elif bits == 16:
        kwargs = dict(device_map='auto')
    else:
        raise NotImplemented

    model = EaModel.from_pretrained(
        base_model_path='NousResearch/Llama-2-7b-chat-hf',
        ea_model_path='yuhuili/EAGLE-llama2-chat-7B',
        #ea_model_path='w32zhong/s3d-EAGLE-retrain-20K',
        torch_dtype=torch.float16,
        random_top_layer=random_top_layer,
        quantize_top_layer=quantize_top_layer,
        **kwargs
    )
    model.eval()

    input_ids = model.tokenizer([prompt], return_tensors="pt").input_ids
    cnt_tokens = 0
    input_ids = input_ids.to('cuda:0')
    past_len = input_ids.shape[1]
    with torch.no_grad():
        for output_ids in model.ea_generate(input_ids, max_length=512):
            #os.system('clear')
            decode_ids = output_ids[0, past_len:].tolist()
            cnt_tokens += len(decode_ids)
            if cnt_tokens >= max_new_tokens:
                break
            past_len = output_ids.shape[1]
            text = model.tokenizer.decode(decode_ids)
            print(text, end=' ', flush=True)
        print()
    return cnt_tokens


def test_vanilla(bits):
    if bits == 4:
        kwargs = dict(load_in_4bit=True, device_map='cuda:0')
    elif bits == 8:
        kwargs = dict(load_in_8bit=True, device_map='cuda:0')
    elif bits == 16:
        kwargs = dict(device_map='auto')
    else:
        raise NotImplemented
    tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-chat-hf')
    inputs = tokenizer([prompt], return_tensors="pt").to('cuda:0')
    model = AutoModelForCausalLM.from_pretrained(
        'NousResearch/Llama-2-7b-chat-hf', trust_remote_code=True,
        torch_dtype=torch.float16, **kwargs
    )
    model.eval()

    streamer = TextStreamer(tokenizer)
    with torch.no_grad():
        output = model.generate(**inputs, streamer=streamer, max_new_tokens=max_new_tokens, use_cache=True)
    cnt_tokens = output.shape[-1] - inputs.input_ids.shape[-1]
    return cnt_tokens


def test(method, bits, random_top_layer, quantize_top_layer):
    print(prompt)
    start_time = time.time()
    if method == 'vanilla':
        cnt_tokens = test_vanilla(bits)
    elif method == 'eagle':
        cnt_tokens = test_eagle(bits,
            random_top_layer=random_top_layer,
            quantize_top_layer=quantize_top_layer
        )
    time_delta = time.time() - start_time
    speed = cnt_tokens / time_delta
    print('e2e speed:', time_delta, cnt_tokens, speed)
    return dict(time_delta=time_delta, cnt_tokens=cnt_tokens, speed=speed)


def mock_test(method, bits, random_top_layer, quantize_top_layer):
    return dict(time_delta=bits, cnt_tokens=2.3, speed=1.2)


if __name__ == '__main__':
    import pandas as pd
    df_params = pd.read_csv('params.tsv', sep='\t', header=0)
    df_results = []
    for params in df_params.to_dict(orient='records'):
        results = test(**params)
        df_results.append(results)
        print(df_results)
    df_results = pd.DataFrame(df_results)
    df_output = pd.concat([df_params, df_results], axis=1)
    print(df_output)
    df_output.to_csv('output.tsv', sep='\t', index=False)
