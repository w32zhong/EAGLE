import torch
from awq.models.base import BaseAWQForCausalLM

import sys
sys.path.insert(0, '.')
from model.ea_model import EaModel

class EagleAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "LlamaDecoderLayer"
    max_seq_len_key = "max_position_embeddings"

    @classmethod
    def from_pretrained(self, **kwargs):
        model = EaModel.from_pretrained(**kwargs)
        model.eval()

        return model.tokenizer, self(model, 'llama',
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

        # attention input
        layers.append(
            dict(
                prev_op=module.input_layernorm,
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


tokenizer, model = EagleAWQForCausalLM.from_pretrained(
    base_model_path='NousResearch/Llama-2-7b-chat-hf',
    ea_model_path='yuhuili/EAGLE-llama2-chat-7B',
    torch_dtype=torch.float16,
    #low_cpu_mem_usage=True,
    use_cache=False
)

quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
model.quantize(tokenizer, quant_config=quant_config)
