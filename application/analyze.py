import torch
from random import randrange
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt

ea_model_path='yuhuili/EAGLE-llama2-chat-7B'
path = hf_hub_download(ea_model_path, 'pytorch_model.bin')
state_dict = torch.load(path)
print(state_dict.keys())

weight, bias = None, None
for key in state_dict.keys():
    val = state_dict[key]
    if key == 'fc.weight':
        weight = val.T
        print(val, val.shape)
    elif key == 'fc.bias':
        bias = val
        print(val, val.shape)

weight
col_idx = randrange(weight.shape[1])
col = weight[:, col_idx].unsqueeze(0).cpu()

plt.imshow( col ) 
plt.show() 
