from liger_kernel.transformers import AutoLigerKernelForCausalLM
import torch
from transformers import AutoTokenizer

def initialize_language_model(self, config):
    self.lm_model = AutoLigerKernelForCausalLM.from_pretrained(
        config.lm_model_name,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_cache=False,
        torch_dtype=torch.bfloat16,
        rope=True,
        swiglu=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
        rms_norm=True
    )
    self.lm_head = self.lm_model.lm_head
    self.lm_head.weight = torch.nn.Parameter(self.lm_model.lm_head.weight.clone())
    self.tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name)
    self.tokenizer.pad_token = self.tokenizer.eos_token
    for param in self.lm_model.model.parameters():
        param.requires_grad = False
