from typing import Optional, Union, Tuple

from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
import torch.nn.functional as F
from mmlm.listener import ListenFeatureExtractor


class MMLM(nn.Module):
    def __init__(self, config="voidful/SmolLM2-360M-Instruct-ASR"):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Language Model Setup
        self.lm_model = AutoModelForCausalLM.from_pretrained(config).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.listener = ListenFeatureExtractor().to(self.device)
        self.adapter = nn.Linear(512, self.lm_model.get_input_embeddings().weight.shape[-1]).to(self.device)
        for param in self.lm_model.parameters():
            param.requires_grad = False

    def encode_text(self, input_ids):
        embeder = self.lm_model.get_input_embeddings()
        # create input_ids and labels for causal language model
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        embedding = embeder(input_ids.to(self.device))
        return embedding, input_ids, labels

    def encode_audio(self, inputs_values):
        inputs_embeds = self.listener(inputs_values.to(self.device))
        inputs_embeds = self.adapter(inputs_embeds.permute(0, 2, 1))
        return inputs_embeds

    def forward(
            self,
            input_values: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.lm_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.lm_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.lm_model.config.use_return_dict

        # encode audio
        audio_embeds = self.encode_audio(input_values)
        text_embeds, input_ids, labels = self.encode_text(labels)

        audio_len = audio_embeds.size(1)
        text_len = text_embeds.size(1)
        if text_len < audio_len:
            padding_size = audio_len - text_len
            text_embeds = F.pad(text_embeds, (0, 0, 0, padding_size))  # (left, right, top, bottom)
        elif text_len > audio_len:
            padding_size = text_len - audio_len
            audio_embeds = F.pad(audio_embeds, (0, 0, 0, padding_size))

        # sum audio and text embeddings
        inputs_embeds = audio_embeds + text_embeds

        # add position embedding

        outputs = self.lm_model(
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs[0]
        loss = None
        if labels is not None:
            logits_len = logits.size(1)
            labels_len = labels.size(1)
            loss_fct = CrossEntropyLoss()
            if labels_len < logits_len:
                padding_size = logits_len - labels_len
                labels = F.pad(labels, (0, padding_size), value=-100)
            elif labels_len > logits_len:
                labels = labels[:, :logits_len]
            shift_logits = logits.reshape(-1, logits.size(-1))
            shift_labels = labels.reshape(-1)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

    def generate(self, input_ids, audio_feature=None, max_length=50):
        self.eval()
        generated = input_ids.to(self.device)
        begin_gen_pos = input_ids.shape[1]
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(input_ids=generated, audio_features=audio_feature)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat((generated, next_token), dim=-1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        return generated[:, begin_gen_pos:-1]
