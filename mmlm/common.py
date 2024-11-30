from liger_kernel.transformers import AutoLigerKernelForCausalLM
import torch
from transformers import AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F


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


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        if logits.dim() > 2:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

        valid_mask = (targets != self.ignore_index)
        logits = logits[valid_mask]
        targets = targets[valid_mask]

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        log_probs_true = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        probs_true = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - probs_true) ** self.gamma
        loss = -focal_weight * log_probs_true
        return loss.mean()
