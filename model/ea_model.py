import copy
import json

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig,AutoConfig
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
#from .modeling_Mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values
from .choices import *
from transformers import AutoTokenizer
import os
from huggingface_hub import hf_hub_download
from .cnets import Model
from .configs import EConfig
from huggingface_hub import hf_hub_download

from .time_stats import TimeStats


class EaModel(nn.Module):

    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            ea_model_path,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path,"r") as f:
            con=json.loads(f.read())
        try:
            bias=con["bias"]
        except:
            bias=True
        self.ea_layer = Model(config,bias=bias)

        low_memory=False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device!=base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                # self.ea_layer.head=nn.Linear(base_model.lm_head.in_features,base_model.lm_head.out_features,bias=False)
                # self.ea_layer.head.weight=copy.deepcopy(base_model.lm_head.weight)
                # self.ea_layer.head.to(device)
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            Type="LLaMA",
            base_model_path=None,
            ea_model_path=None,
            ### newly added ###
            random_top_layer=False,
            quantize_top_layer=False,
            ###################
            **kwargs,
    ):
        #assert Type=="LLaMA" or "Mixtral"
        Type=AutoConfig.from_pretrained(base_model_path).architectures[0]
        if Type=='LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            raise NotImplemented
            #base_model = KVMixtralForCausalLM.from_pretrained(
            #    base_model_path, **kwargs
            #)

        configpath=os.path.join(ea_model_path,"config.json")
        if not os.path.exists(configpath):
            try:
                configpath = hf_hub_download(ea_model_path, "config.json")
            except:
                configpath = './fallback_config.json'
        model = cls(
            base_model,
            base_model_path,
            configpath
        )
        load_model_path=os.path.join(ea_model_path, "pytorch_model.bin")
        ea_layer_state_dict = None
        if not os.path.exists(load_model_path):
            try:
                load_model_path=hf_hub_download(ea_model_path, "pytorch_model.bin")
            except:
                from safetensors.torch import load_file
                load_model_path=hf_hub_download(ea_model_path, "model.safetensors")
                ea_layer_state_dict = load_file(load_model_path, device='cuda:0')

        if ea_layer_state_dict is None:
            ea_layer_state_dict = torch.load(load_model_path, map_location=base_model.device)

        import bitsandbytes as bnb
        #for key in ea_layer_state_dict:
        #    ea_layer_state_dict[key] = ea_layer_state_dict[key].to(dtype=torch.float16)
        to_change = []
        if quantize_top_layer:
            for key, module in model.ea_layer.named_modules():
                if not isinstance(module, torch.nn.Linear):
                    continue
                if key == 'fc':
                    continue
                path_fields = key.split('.')
                parent_key = '.'.join(path_fields[:-1])
                child_key = path_fields[-1]
                parent = model.ea_layer.get_submodule(parent_key)
                to_change.append((key, module, parent, child_key))
            #print(model.ea_layer.layers[0].mlp.gate_proj.weight)
            for key, module, parent, child_key in to_change:
                delattr(parent, child_key)
                cls = bnb.nn.LinearNF4 # Linear4bit
                if kwargs.get('load_in_8bit'):
                    cls = bnb.nn.Linear8bitLt
                elif kwargs.get('load_in_4bit'):
                    cls = bnb.nn.LinearNF4
                else:
                    pass
                print(key, cls)
                setattr(parent, child_key,
                    cls(module.in_features, module.out_features, bias=False)
                )

        if not random_top_layer:
            print('loading top-layer state dict ...')
            model.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
            print(model.ea_layer)

        model.ea_layer.to(model.ea_layer.fc.weight.device)
        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
            init=True,
            logits_processor=None
    ):
        with torch.inference_mode():
            # Pass input through the base model
            assert attention_mask is None
            #print(past_key_values[0][0].shape, past_key_values[0][0].data.device)
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0].clone()

        if init:
            if logits_processor is not None:
                logits = orig[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                token = torch.multinomial(probabilities, 1)
            else:
                token = torch.argmax(orig[:, -1])
                token = token[None, None]

            # concate input_ids with an initial verified input_id
            input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

            # hidden_states: [B, L, 4096]
            # input_ids: [B, L+1]
            # topK_genrate_res: (torch.cat(ss_token),torch.cat(ss_prob),ss_op)
            topK_genrate_res = self.ea_layer.topK_genrate(hidden_states, input_ids, self.base_model.lm_head, logits_processor)
            # topK_genrate_res[0]: [1+4+4+1+1=11, topk=10] where 11 is the draft token tree non-leaf size

            #interact
            #self.tokenizer.decode(input_ids[0])
            #for i in range(len(topK_genrate_res[0])):
            #    print(i, [self.tokenizer.decode([t]) for t in topK_genrate_res[0][i]])

            if output_orig:
                return topK_genrate_res, outputs, orig, hidden_states, token
            return topK_genrate_res, hidden_states, token
        else:
            if output_orig:
                return outputs, orig, hidden_states

#    @torch.no_grad()
#    def eagenerate(
#            self,
#            input_ids,
#            temperature=0.0,
#            top_p=0.0,
#            top_k=0.0,
#            max_new_tokens=512,
#            max_length=2048,
#            tree_choices=mc_sim_7b_63,
#
#    ):
#        if temperature > 1e-5:
#            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
#        else:
#            logits_processor = None
#        #assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
#        # Avoid modifying the input_ids in-place
#        input_ids = input_ids.clone()
#        self.ea_layer.reset_kv()
#
#        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
#            tree_buffers = self.tree_buffers
#        else:
#            tree_buffers = generate_tree_buffers(
#                tree_choices, device=self.base_model.model.layers[-1].self_attn.q_proj.weight.device
#            )
#            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
#                self.base_model.lm_head.weight.device)
#        self.tree_buffers = tree_buffers
#        self.tree_choices = tree_choices
#
#        # Initialize the past key and value states
#        if hasattr(self, "past_key_values"):
#            past_key_values = self.past_key_values
#            past_key_values_data = self.past_key_values_data
#            current_length_data = self.current_length_data
#            # Reset the past key and value states
#            current_length_data.zero_()
#        else:
#            (
#                past_key_values,
#                past_key_values_data,
#                current_length_data,
#            ) = initialize_past_key_values(self.base_model)
#            self.past_key_values = past_key_values
#            self.past_key_values_data = past_key_values_data
#            self.current_length_data = current_length_data
#
#        input_len = input_ids.shape[1]
#        reset_tree_mode(self)
#        tree_logits, logits, hidden_state, sample_token = initialize_tree(
#            input_ids, self, tree_buffers["tree_attn_mask"], past_key_values, logits_processor
#        )
#        new_token = 0
#
#        for idx in range(max_length):
#            candidates, cart_candidates_prob, tree_candidates = generate_candidates(
#                tree_logits,
#                tree_buffers["tree_indices"],
#                tree_buffers["retrieve_indices"],
#                sample_token,
#                logits_processor
#            )
#            logits, hidden_state_new, outputs = tree_decoding(
#                self,
#                tree_candidates,
#                past_key_values,
#                tree_buffers["tree_position_ids"],
#                input_ids,
#                tree_buffers["retrieve_indices_head"],
#            )
#            best_candidate, accept_length, sample_p = evaluate_posterior(
#                logits, candidates, logits_processor, cart_candidates_prob, tree_logits[2], tree_buffers["p_indices"],
#                tree_candidates, tree_buffers["b_indices"]
#            )
#            input_ids, tree_logits, new_token, hidden_state, sample_token = update_inference_inputs(
#                input_ids,
#                candidates,
#                best_candidate,
#                accept_length,
#                tree_buffers["retrieve_indices"],
#                logits_processor,
#                logits,
#                tree_logits,
#                new_token,
#                past_key_values_data,
#                current_length_data,
#                self,
#                hidden_state,
#                hidden_state_new,
#                sample_p
#            )
#
#            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
#                return input_ids
#            if new_token > max_new_tokens:
#                return input_ids
#            if input_ids.shape[1] > max_length:
#                return input_ids

    @torch.no_grad()
    def ea_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_length=512,
            #tree_choices=mc_sim_d1, # 11
            #tree_choices=mc_sim_d2, # 14
            #tree_choices=mc_sim_d3, # 16
            tree_choices=mc_sim_7b_63, # 16
            profile_acceptance_rate=False
    ):
        # n_params = sum(x.numel() for x in self.base_model.parameters())
        # n_params = n_params / (1_000 * 1_000 * 1_000)
        # print(n_params)
        # n_params = sum(x.numel() for x in self.ea_layer.parameters())
        # n_params = n_params / (1_000 * 1_000 * 1_000)
        # print(n_params)
        # ratio: 0.054

        time_stats = TimeStats()
        time_stats.start('ea_generate init')

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
            tree_buffers = self.tree_buffers
        else:
            # construct buffer for tree-attention
            tree_buffers = generate_tree_buffers(
                tree_choices, device=self.base_model.model.layers[-1].self_attn.q_proj.weight.device
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                self.base_model.lm_head.weight.device)
        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices

        # tree_buffers.keys():
        [
            'tree_attn_mask', # torch.Size([1, 1, 26, 26])
            'tree_indices', # torch.Size([26]), i.e., node idx -> node id @ 10-degree binary tree
            'tree_position_ids', # torch.Size([26]), i.e., node idx -> node depth
            'retrieve_indices', # torch.Size([15, 6]), i.e., tree_indices[retrieve_indices] = leaf-root paths
            'p_indices',
            'b_indices',
            'retrieve_indices_head' # torch.Size([15, 6])
        ]

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)

        # initialize_tree(init=True, output_orig=True)
        # returns what EaModel::forward() returns:
        # return ea_logits, ..., orig, hidden_states, token
        # (except the 2nd value)
        # where ea_logits/tree_logits is returned from cnets::topK_genrate()
        # which returns (torch.cat(ss_token), torch.cat(ss_prob), ss_op)
        tree_logits, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, tree_buffers["tree_attn_mask"], past_key_values, logits_processor
        )
        # input_ids: [1, L]
        # tree_logits: ([11, top10], [11, top10], [None, None, None, None, None])
        # logits: [1, L, 32000]
        # hidden_state: [1, L, 4096]
        # sample_token: [1, 1]
        new_token = 0

        time_stats.stop('ea_generate init')

        for idx in range(max_length):
            time_stats.start('ea_generate iteration')
            candidates, cart_candidates_prob, tree_candidates = generate_candidates(
                tree_logits,
                tree_buffers["tree_indices"],
                tree_buffers["retrieve_indices"],
                sample_token,
                logits_processor
            )
            # candidates of size [15, 6] represents possible leaf-root paths
            # tree_candidates of size [26], represents the filled draft tree

            # p self.tokenizer.decode(sample_token[0])
            # p self.tokenizer.decode(tree_candidates[0])
            # p self.tokenizer.batch_decode(candidates)

            # verify-forward!
            # initialize_tree(init=False, output_orig=True)
            # which returns (outputs, orig, hidden_states)

            cur_len, past_len = tree_candidates.shape[-1], past_key_values[0][0].current_length
            time_stats.push('cur_len', cur_len)
            time_stats.push('past_len', past_len.item())

            time_stats.start('ea_generate verify forward')
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                tree_buffers["tree_position_ids"],
                input_ids,
                tree_buffers["retrieve_indices_head"],
            )
            time_stats.stop('ea_generate verify forward')
            # p self.tokenizer.batch_decode(logits[:,:-1].argmax(-1))

            # verify evaluate!
            time_stats.start('ea_generate verify eval')
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor, cart_candidates_prob, tree_logits[2], tree_buffers["p_indices"],
                tree_candidates, tree_buffers["b_indices"],
                time_stats, tree_buffers["retrieve_indices"],
                profile_acceptance_rate=profile_acceptance_rate
            )
            time_stats.stop('ea_generate verify eval')
            time_stats.push('#new tokens per iteration', accept_length.item() + 1)

            last_input_ids = input_ids
            # calling cnets::topK_genrate() in update_inference_inputs (model/utils.py)
            input_ids, tree_logits, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                tree_buffers["retrieve_indices"],
                logits_processor,
                logits,
                tree_logits,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state,
                hidden_state_new,
                sample_p,
                time_stats
            )
            time_stats.stop('ea_generate iteration')

            #print('new tokens:', accept_length.item() + 1)
            #print('output tokens:', input_ids.shape[1] - last_input_ids.shape[1])
            assert accept_length.item() + 1 == input_ids.shape[1] - last_input_ids.shape[1]

            yield input_ids

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                print(time_stats.report())
                self.model_stats = time_stats._hist
                break
            if input_ids.shape[1] >= max_length:
                print(time_stats.report())
                self.model_stats = time_stats._hist
                break

    def lce_model_stats(self):
        return self.model_stats

    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_steps=512,
            tree_choices=mc_sim_7b_63,

    ):
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
            tree_buffers = self.tree_buffers
        else:
            tree_buffers = generate_tree_buffers(
                tree_choices, device=self.base_model.model.layers[-1].self_attn.q_proj.weight.device
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                self.base_model.lm_head.weight.device)
        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0

        for idx in range(max_steps):
            input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)

            yield input_ids

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > 1024:
                break
            if input_ids.shape[1] > 1960:
                break
