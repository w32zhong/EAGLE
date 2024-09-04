import torch
from awq.quantize import pre_quant

import sys
sys.path.insert(0, '.')
from model.ea_model import EaModel

model = EaModel.from_pretrained(
    base_model_path='NousResearch/Llama-2-7b-chat-hf',
    ea_model_path='yuhuili/EAGLE-llama2-chat-7B',
    torch_dtype=torch.float16,
)

tokenizer = model.tokenizer

q_config = {
    "zero_point": True,
    "q_group_size": 128,
}

run_awq(model, tokenizer, 4, q_config,
    n_samples=512, seqlen=512,
    auto_scale=True, mse_range=True,
    calib_data="pileval",
)
