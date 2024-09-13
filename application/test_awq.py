import os
import re
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer
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
use_original = False

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


    def save_quantized(self, model_to_save, save_pth):
        Q_state_dict = dict()
        for path, module in model_to_save.named_modules():
            name = module.__class__.__name__
            print('saving', path, name)
            if 'WQLinear' not in name:
                continue
            module_state_dict = module.state_dict()
            for key in module_state_dict:
                Q_state_dict[path + f".{key}"] = module
        torch.save(Q_state_dict, save_pth)

    @staticmethod
    def get_model_layers(model):
        layers = model.base_model.model.layers + model.ea_layer.layers
        return layers

    @staticmethod
    def get_act_for_scaling(module):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model, device: str):
        model.base_model.model.embed_tokens.to(device)
        model.ea_layer.embed_tokens.to(device)

    @staticmethod
    def get_layers_for_scaling(module, input_feat, module_kwargs):
        layers = []

        # dict(
        #     prev_op: the previous operator
        #     layers: linear weights to concate
        #     inp: inputs of this operator
        #     module2inspect: module to call forward(inp, kwargs)
        # )

        # attention input
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

        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )

        # linear 1
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )

        # linear 2
        layers.append(
            dict(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )

        return layers

    def quantize_layer(self, layer, input_feat, quant_config):
        module_config = self.get_layers_for_scaling(layer, input_feat, {})
        self.quantize(None, quant_config=quant_config, init_only=True)
        scales_list = [
            self.quantizer._search_best_scale(idx, layer, **linear_config)
            for idx, linear_config in enumerate(module_config)
        ]
        apply_scale(layer, scales_list, input_feat_dict=input_feat)
        clear_memory()

        named_linears = {
            path: m for path, m in layer.named_modules()
            if isinstance(m, nn.Linear)
        }
        clip_list = self.quantizer._search_best_clip(
            layer, named_linears, input_feat
        )
        apply_clip(layer, clip_list)
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


def quantize(save_dir='save'):
    tokenizer, ea_model, awq_model = EagleAWQForCausalLM.from_pretrained(
        base_model_path='NousResearch/Llama-2-7b-chat-hf',
        ea_model_path='yuhuili/EAGLE-llama2-chat-7B',
        torch_dtype=torch.float16,
        device_map=('auto' if not use_original else None)
    )
    tokenizer = ea_model.tokenizer

    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

    if use_original:
        awq_model.quantize(tokenizer, quant_config=quant_config)
        quit()

    input_feat = defaultdict(list)
    with AWQCalibration(awq_model, '', input_feat), torch.no_grad():
        prompt = '[INST] tell me something interesting about the solar eclipse in April 2024. [/INST]'
        input_ids = tokenizer([prompt], return_tensors="pt").input_ids
        input_ids = input_ids.to('cuda:0')
        cnt_tokens = 0
        past_len = input_ids.shape[1]
        for output_ids in ea_model.ea_generate(input_ids, max_length=128):
            decode_ids = output_ids[0, past_len:].tolist()
            cnt_tokens += len(decode_ids)
            past_len = output_ids.shape[1]
            text = tokenizer.decode(decode_ids)
            print(text, end=' ', flush=True)
        print()
    input_feat = {k: torch.cat(v, dim=1) for k, v in input_feat.items()}
    clear_memory()

    all_layers = input_feat.keys()
    all_layers = [re.sub('(layers\.\d+)\..*', '\g<1>', s) for s in all_layers]
    all_layers = set(all_layers)

    for i, layer_path in enumerate(all_layers):
        print(f'Quantizing {i}-th layer:', layer_path)
        layer = awq_model.get_submodule(layer_path)
        layer_input_feat = {
            k.replace(layer_path + '.', ''): v
            for k, v in input_feat.items()
            if k.startswith(layer_path)
        }
        awq_model.quantize_layer(layer, layer_input_feat, quant_config)
        clear_memory()

    os.makedirs(save_dir, exist_ok=True)
    awq_model.save_quantized(ea_model, f'{save_dir}/save.pth')


def quantize_vanilla(model_path='NousResearch/Llama-2-7b-chat-hf', save_dir='save'):
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(
        model_path, device_map="auto", use_cache=False
    )
    model.quantize(tokenizer, quant_config=quant_config)
    model.save_quantized(save_dir)


def load_and_test(mode, pth_path='save/save.pth'):
    if mode == 'fp16':
        kwargs = dict(quantize_top_layer=False, load_in_4bit=False)
    elif mode == 'int8':
        kwargs = dict(quantize_top_layer=True, load_in_8bit=True)
    elif mode == 'nf4':
        kwargs = dict(quantize_top_layer=True, load_in_4bit=True)
    elif mode == 'nf4-baseonly':
        kwargs = dict(quantize_top_layer=False, load_in_4bit=True)
    elif mode == 'nf4-toponly':
        kwargs = dict(quantize_top_layer=True)
    elif mode == 'nf4-awq':
        kwargs = dict(quantize_top_layer=False, load_in_4bit=True)
    elif mode == 'awq':
        kwargs = dict()
    else:
        raise ValueError
    tokenizer, ea_model, awq_model = EagleAWQForCausalLM.from_pretrained(
        base_model_path='NousResearch/Llama-2-7b-chat-hf',
        ea_model_path='yuhuili/EAGLE-llama2-chat-7B',
        torch_dtype=torch.float16,
        device_map=('auto' if not use_original else None),
        ########
        **kwargs
    )
    tokenizer = ea_model.tokenizer

    if 'awq' in mode:
        quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
        quant_config = AwqConfig.from_dict(quant_config)
        Q_state_dict = torch.load(pth_path)
        to_be_replaced = []
        for key, module in ea_model.named_modules():
            def parent(key):
                path_fields = key.split('.')
                parent_key = '.'.join(path_fields[:-1])
                child_key = path_fields[-1]
                return parent_key, child_key
            Q_mod_keys = set([parent(key)[0] for key in Q_state_dict.keys()])

            if key not in Q_mod_keys:
                continue

            version = quant_config.version
            if version == "gemm":
                q_linear_module = WQLinear_GEMM
            elif version == "gemv":
                q_linear_module = WQLinear_GEMV
            elif version == "gemv_fast":
                q_linear_module = WQLinear_GEMVFast
            else:
                q_linear_module = None

            q_linear = q_linear_module.from_linear(
                module, quant_config.w_bit, quant_config.q_group_size, True
            )
            for subkey, dst_W in q_linear.named_buffers():
                state = Q_state_dict[key + '.' + subkey]
                src_W = getattr(state, subkey)
                dst_W.copy_(src_W)
            q_linear.to(next(module.parameters()).device)

            parent_key, child_key = parent(key)
            p_module = ea_model.get_submodule(parent_key)

            to_be_replaced.append((p_module, child_key, q_linear))

        for p_module, child_key, _ in to_be_replaced:
            delattr(p_module, child_key)
        clear_memory()
        for p_module, child_key, q_linear in to_be_replaced:
            q_linear.weight = q_linear.qweight # proxy for compability
            setattr(p_module, child_key, q_linear)

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


if __name__ == '__main__':
                                        # vinilla_speed=15
    #load_and_test('fp16')              # speed=42.5
    #load_and_test('int8')              # speed=9.6
    #load_and_test('nf4')               # speed=5.3

    #load_and_test('nf4-baseonly')      # speed=8.7
    #load_and_test('nf4-toponly')       # speed=19.8   ***
    #quantize()
    #load_and_test('nf4-awq')           # speed=6.6
    #load_and_test('fp16-awq', 'save/model.ea_layer.layers.0.pth')          # speed=31.2   ***

    #load_and_test('awq')
    quantize_vanilla()
