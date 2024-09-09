import torch
import torch.nn as nn
from awq.models.base import BaseAWQForCausalLM
from awq.quantize.scale import apply_scale, apply_clip
from collections import defaultdict
from functools import partial

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
                prev_op=module.input_layernorm if use_original else None,
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

class AWQCalibration():
    def __init__(self, model, target_path, input_feat):
        self.model = model
        self.target_path = target_path
        self.input_feat = input_feat
        self.hooks = []

    @staticmethod
    def hook_fn(self, path, module, inputs, kwargs, output):
        x = inputs[0].detach().cpu()
        short_path = path.replace(self.target_path, '')
        self.input_feat[short_path].append(x)

    def __enter__(self):
        for path, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if path.startswith(self.target_path):
                print('[hook]', path)
                hook = module.register_forward_hook(
                    partial(self.hook_fn, self, path),
                    with_kwargs=True
                )
                self.hooks.append(hook)

    def __exit__(self, type, value, traceback):
        for hook in self.hooks:
            hook.remove()


input_feat = defaultdict(list)
with AWQCalibration(awq_model, 'model.ea_layer.layers.0.', input_feat), torch.no_grad():
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
module = ea_model.ea_layer.layers[0]
awq_model.quantize_layer(module, input_feat, quant_config)
