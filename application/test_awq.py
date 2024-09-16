import os
import re
import time
import torch
import json
import pickle
import torch.nn as nn
from transformers import AutoTokenizer, TextStreamer
from awq import AutoAWQForCausalLM
from awq.models.base import BaseAWQForCausalLM
from awq.quantize.scale import apply_scale, apply_clip
from awq.utils.utils import clear_memory
from collections import defaultdict
from functools import partial
from awq.models._config import AwqConfig
from awq.modules.linear import (
    WQLinear_GEMM,
    WQLinear_GEMV,
    WQLinear_GEMVFast,
)

import sys
sys.path.insert(0, '.')
from model.ea_model import EaModel


def test_vanilla(model_path='NousResearch/Llama-2-7b-chat-hf', save_dir='save'):
    prompt = '[INST] tell me something interesting about the solar eclipse in April 2024. [/INST]'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if save_dir is None: # inference
        model = AutoAWQForCausalLM.from_quantized('save/', fuse_layers=False)
        inputs = tokenizer([prompt], return_tensors="pt").to('cuda:0')
        model.eval()
        print(model)
        streamer = TextStreamer(tokenizer)
        with torch.no_grad():
            output = model.generate(**inputs, streamer=streamer, max_length=128)
    else:
        quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
        model = AutoAWQForCausalLM.from_pretrained(
            model_path, device_map="auto", use_cache=False
        )
        model.quantize(tokenizer, quant_config=quant_config, apply_clip=False)
        model.save_quantized(save_dir)


class EagleAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "LlamaDecoderLayer"
    max_seq_len_key = "max_position_embeddings"

    @classmethod
    def from_pretrained(self, **kwargs):
        model = EaModel.from_pretrained(**kwargs)
        model.eval()

        return model.tokenizer, model, self(model, 'llama',
            is_quantized=False, config=model.config,
            quant_config=None, processor=None)


    def save_quantized(self, layers, save_pth):
        Q_state_dict = dict()
        for layer_path in layers:
            layer = self.get_submodule(layer_path)
            layer_config = self.get_layers_for_scaling(layer, None, None)
            layer_prev_ops = [c['prev_op'] for c in layer_config]
            to_save = []
            for subkey, submod in layer.named_modules():
                name = submod.__class__.__name__.lower()
                if 'wqlinear' in name or submod in layer_prev_ops:
                    sd = submod.state_dict()
                    for state_key in sd:
                        key = f'{layer_path}.{subkey}.{state_key}'
                        Q_state_dict[key] = sd[state_key]
        torch.save(Q_state_dict, save_pth)

    @staticmethod
    def get_model_layers(model):
        layers = model.base_model.model.layers + model.ea_layer.layers
        return layers

    @staticmethod
    def get_layers_for_scaling(module, input_feat, module_kwargs):
        layers = []
        if input_feat is None:
            input_feat = defaultdict(int)

        layers.append(
            dict(
                prev_op=module.input_layernorm if hasattr(module, 'input_layernorm') else module.E,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        layers.append(
            dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat["self_attn.o_proj"],
            )
        )
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )
        layers.append(
            dict(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )

        return layers

    def quantize_layer(self, layer, input_feat, quant_config):
        # initialize
        module_config = self.get_layers_for_scaling(layer, input_feat, {})
        self.quantize(None, quant_config=quant_config, init_only=True)
        named_linears = {
            path: m for path, m in layer.named_modules()
            if isinstance(m, nn.Linear)
        }
        # find and apply scaling factors
        scales_list = [
            self.quantizer._search_best_scale(idx, layer, **linear_config)
            for idx, linear_config in enumerate(module_config)
        ]
        apply_scale(layer, scales_list, input_feat_dict=input_feat)
        clear_memory()
        # find and apply clipping thresholds
        clip_list = self.quantizer._search_best_clip(
            layer, named_linears, input_feat
        )
        apply_clip(layer, clip_list)
        clear_memory()
        # quantize
        self.quantizer._apply_quant(layer, named_linears)
        clear_memory()


class AWQCalibration():
    def __init__(self, model, target_path, input_feat):
        self.model = model
        self.target_path = target_path + '.' if target_path else None
        self.input_feat = input_feat
        self.hooks = []

    @staticmethod
    def hook_fn(self, path, module, inputs, kwargs, output):
        x = inputs[0].detach().cpu()
        self.input_feat[path].append(x)

    def __enter__(self):
        for path, module in self.model.named_modules():
            if not isinstance(module, nn.Linear) or 'layers' not in path:
                continue
            if self.target_path is None:
                short_path = path
            else:
                short_path = path.replace(self.target_path, '')
            if self.target_path is None or path.startswith(self.target_path):
                print('[hook]', path)
                hook = module.register_forward_hook(
                    partial(self.hook_fn, self, short_path),
                    with_kwargs=True
                )
                self.hooks.append(hook)

    def __exit__(self, type, value, traceback):
        for hook in self.hooks:
            hook.remove()


def create_calib_files(tokenizer, awq_model, ea_model, save_dir):
    with open('application/prompts.json', 'r') as fh:
        prompts = json.load(fh)
    calib_questions = [p['prompt_text'] for p in prompts][:]

    all_layer_paths = [f'model.base_model.model.layers.{l}' for l in range(32)] + ['model.ea_layer.layers.0']

    for m, layer_path in enumerate(all_layer_paths):
        input_feat = defaultdict(list)
        with AWQCalibration(awq_model, layer_path, input_feat), torch.no_grad():
            for n, prompt in enumerate(calib_questions):
                prompt = '[INST] ' + prompt + ' [/INST]'
                print(m, len(all_layer_paths), layer_path)
                print(n, len(calib_questions), prompt)
                input_ids = tokenizer([prompt], return_tensors="pt").input_ids
                input_ids = input_ids.to('cuda:0')
                cnt_tokens = 0
                past_len = input_ids.shape[1]
                for output_ids in ea_model.ea_generate(input_ids, max_length=512):
                    decode_ids = output_ids[0, past_len:].tolist()
                    cnt_tokens += len(decode_ids)
                    past_len = output_ids.shape[1]
                    text = tokenizer.decode(decode_ids)
                    print(text, end=' ', flush=True)
                print()
        with open(f'{save_dir}/{layer_path}.pkl', 'wb') as fh:
            input_feat = {k: torch.cat(v, dim=1) for k, v in input_feat.items()}
            # factory calib data: torch.Size([65, 512, 4096])
            # p input_feat.popitem()[1].shape
            pickle.dump(input_feat, fh)
        clear_memory()


def AWQ_quantize(slice_layers, quant_config, awq_model):
    group_size = quant_config['q_group_size']
    fname = f'awq-layers({slice_layers.start}-{slice_layers.stop}-{slice_layers.step})-g{group_size}'
    save_pth = f'save/{fname}.pth'
    os.makedirs('save', exist_ok=True)
    if os.path.exists(save_pth):
        return save_pth
    all_layer_paths = [f'model.base_model.model.layers.{l}' for l in range(32)] + ['model.ea_layer.layers.0']
    print('Targeting layers:', all_layer_paths[slice_layers])
    for i, layer_path in enumerate(all_layer_paths[slice_layers]):
        print(f'Quantizing {i}-th layer:', layer_path)
        with open(f'./save/calib/{layer_path}.pkl', 'rb') as fh:
            input_feat = pickle.load(fh)
        for key, feat in input_feat.items():
            # in original version this is 65 * 512 * 4096
            batch = min(feat.shape[1] // 512, 30)
            extra = feat.shape[1] % 512
            dim = feat.shape[-1]
            feat = feat[:,:-extra,:].reshape(-1, 512, dim)
            feat = feat[:batch, ...]
            input_feat[key] = feat
            #print(key, feat.shape)
        clear_memory()
        layer = awq_model.get_submodule(layer_path)
        awq_model.quantize_layer(layer, input_feat, quant_config)
        clear_memory()

    awq_model.save_quantized(all_layer_paths[slice_layers], save_pth)
    return save_pth


def test(comment, quantize_top_layer=False, load_in_8bit=False, load_in_4bit=False,
        awq_group_size=-1, awq_layers=None, awq_kernel="GEMM", results=None):

    tokenizer, ea_model, awq_model = EagleAWQForCausalLM.from_pretrained(
        base_model_path='NousResearch/Llama-2-7b-chat-hf',
        ea_model_path='yuhuili/EAGLE-llama2-chat-7B',
        torch_dtype=torch.float16,
        device_map='auto',
        ########
        quantize_top_layer=quantize_top_layer,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit
    )
    tokenizer = ea_model.tokenizer
    #create_calib_files(tokenizer, awq_model, ea_model, './save/calib')

    def parent(key):
        path_fields = key.split('.')
        parent_key = '.'.join(path_fields[:-1])
        child_key = path_fields[-1]
        return parent_key, child_key

    if awq_layers:
        quant_config = {
            "zero_point": True,
            "q_group_size": awq_group_size,
            "w_bit": 4,
            "version": awq_kernel
        }
        awq_layers = eval(awq_layers)
        Q_state_dict = torch.load(AWQ_quantize(awq_layers, quant_config, awq_model))
        del tokenizer, ea_model, awq_model
        clear_memory()
        clear_memory()
        breakpoint()
        tokenizer, ea_model, awq_model = EagleAWQForCausalLM.from_pretrained(
            base_model_path='NousResearch/Llama-2-7b-chat-hf',
            ea_model_path='yuhuili/EAGLE-llama2-chat-7B',
            torch_dtype=torch.float16,
            device_map='auto',
            quantize_top_layer=quantize_top_layer,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )
        quant_config = AwqConfig.from_dict(quant_config)
        Q_mod_keys = set([parent(key)[0] for key in Q_state_dict.keys()])
        to_be_replaced = []
        for key, module in awq_model.named_modules():
            if key not in Q_mod_keys:
                continue
            elif 'layernorm' in key:
                continue

            version = quant_config.version
            if version == "gemm":
                q_linear_module = WQLinear_GEMM
            elif version == "gemv":
                q_linear_module = WQLinear_GEMV
            elif version == "gemv_fast":
                q_linear_module = WQLinear_GEMVFast
            else:
                raise NotImplemented

            q_linear = q_linear_module.from_linear(
                module, quant_config.w_bit, quant_config.q_group_size, True
            )
            q_linear.to(next(module.parameters()).device)

            parent_key, child_key = parent(key)
            p_module = awq_model.get_submodule(parent_key)

            to_be_replaced.append((p_module, child_key, q_linear))

        for p_module, child_key, _ in to_be_replaced:
            #save = getattr(p_module, child_key)
            delattr(p_module, child_key)
            #setattr(p_module, child_key + '_original', save)
        clear_memory()
        for p_module, child_key, q_linear in to_be_replaced:
            q_linear.weight = q_linear.qweight # proxy for compability
            setattr(p_module, child_key, q_linear)

        awq_model.load_state_dict(Q_state_dict, strict=False)

    print(ea_model)
    with torch.no_grad():
        prompt = '[INST] tell me something interesting about the solar eclipse in April 2024. [/INST]'
        input_ids = tokenizer([prompt], return_tensors="pt").input_ids
        input_ids = input_ids.to('cuda:0')
        cnt_tokens = 0
        past_len = input_ids.shape[1]

        start_time = time.time()
        for output_ids in ea_model.ea_generate(input_ids, max_length=512): #128
            decode_ids = output_ids[0, past_len:].tolist()
            cnt_tokens += len(decode_ids)
            past_len = output_ids.shape[1]
            text = tokenizer.decode(decode_ids)
            print(text, end=' ', flush=True)
        print()
        time_delta = time.time() - start_time
    speed = cnt_tokens / time_delta
    print('e2e speed:', time_delta, cnt_tokens, speed)
    results['time_delta'] = time_delta
    results['cnt_tokens'] = cnt_tokens
    results['speed'] = speed


if __name__ == '__main__':
    import os
    import argparse
    import pandas as pd
    import multiprocessing, threading
    from colorama import Fore, Back, Style
    parser = argparse.ArgumentParser(description='Pandas Fire ArgumentParser')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    manager = multiprocessing.Manager()
    df_params = pd.read_csv('params-awq.tsv', sep='\t', header=0)
    df_params = df_params.replace({float('nan'): None})
    df_results = []
    for params in df_params.to_dict(orient='records'):
        print(Fore.RED, Back.YELLOW, params, Style.RESET_ALL)
        try:
            params['results'] = manager.dict()
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            if args.debug:
                process = threading.Thread(target=test, kwargs=params)
            else:
                process = multiprocessing.Process(target=test, kwargs=params)
            process.start()
            process.join()
        except:
            if args.debug:
                pass
            else:
                process.terminate()
            break
        results = dict(params['results'])
        print(Fore.RED, Back.YELLOW, results, Style.RESET_ALL, end='\n\n')
        df_results.append(results)
    df_results = pd.DataFrame(df_results)
    df_output = df_params.join(df_results)
    print(df_output)
    df_output.to_csv('output-awq.tsv', sep='\t', index=False)
