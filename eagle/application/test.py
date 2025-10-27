import torch
import transformers
from eagle.model.ea_model import EaModel
print(transformers.__path__)

model = EaModel.from_pretrained(
    base_model_path='meta-llama/Llama-2-7b-chat-hf',
    ea_model_path='yuhuili/EAGLE-llama2-chat-7B',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    load_in_4bit=False,
    load_in_8bit=False,
    device_map="cuda",
    use_eagle3=False,
)

breakpoint()
del model.ea_layer.embed_tokens
torch.cuda.empty_cache()
breakpoint()

model.eval()
