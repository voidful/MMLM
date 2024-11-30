import os

from mmlm.common import initialize_language_model, FocalLoss

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import Optional, Union, Tuple
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.configuration_utils import PretrainedConfig
from mmlm.listener import ListenFeatureExtractor
from mmlm.utility import align_and_sum_embeddings, prepare_labels


class MMLMASRConfig(PretrainedConfig):
    model_type = "mmlmasr"

    def __init__(self, lm_model_name="voidful/SmolLM2-360M-Instruct-Whisper", **kwargs):
        super().__init__(**kwargs)
        self.lm_model_name = lm_model_name


class MMLMASR(PreTrainedModel):
    config_class = MMLMASRConfig

    def __init__(self, config: MMLMASRConfig):
        super().__init__(config)
        self.config = config
        initialize_language_model(self, config)
        self._initialize_asr_components()

    def _initialize_asr_components(self):
        self.listener = ListenFeatureExtractor()
        embedding_dim = self.lm_model.get_input_embeddings().weight.size(-1)
        self.adapter = nn.Linear(512, embedding_dim)

    def embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeder = self.lm_model.get_input_embeddings()
        return embeder(input_ids)

    def embed_audio(self, input_values: torch.Tensor) -> torch.Tensor:
        features = self.listener(input_values)
        adapted_features = self.adapter(features.permute(0, 2, 1))
        return adapted_features

    def embed_system_prompt(self,
                            prompt_text: str = "you are a helpful asr model, please help me to understand the audio.") -> torch.Tensor:
        template = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": f"{prompt_text}"}],
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        )
        system_emb = self.embed_text(template.to(self.lm_model.device))
        return system_emb

    def forward(
            self,
            input_values: Optional[torch.FloatTensor] = None,
            asr_texts: Optional[torch.LongTensor] = None,
            asr_labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        asr_texts, asr_labels = prepare_labels(self.tokenizer, asr_texts, asr_labels)

        # Encode inputs
        audio_embeds = self.embed_audio(input_values)
        text_embeds = self.embed_text(asr_texts)
        inputs_embeds = align_and_sum_embeddings(audio_embeds, text_embeds)
        system_embeds = self.embed_system_prompt()
        outputs = self.lm_model(
            inputs_embeds=torch.cat([system_embeds, inputs_embeds], dim=1),
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoded_labels = F.pad(asr_labels, (system_embeds.shape[1], 0), value=-100)
        loss = self._compute_loss(outputs.logits, encoded_labels)

        if not return_dict:
            output = (outputs.logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.FloatTensor:
        logits_len, labels_len = logits.size(1), labels.size(1)
        if labels_len < logits_len:
            labels = F.pad(labels, (0, logits_len - labels_len), value=-100)
        elif labels_len > logits_len:
            labels = labels[:, :logits_len]

        loss_fct = FocalLoss()
        shift_logits = logits.reshape(-1, logits.size(-1))
        shift_labels = labels.reshape(-1)
        return loss_fct(shift_logits, shift_labels)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.lm_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.lm_model.gradient_checkpointing_disable()

    def generate(self, input_ids: torch.Tensor, audio_feature: Optional[torch.Tensor] = None, max_length: int = 50):
        self.eval()
        generated = input_ids
        begin_gen_pos = input_ids.size(1)

        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(input_values=audio_feature, labels=generated)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        return generated[:, begin_gen_pos:-1]
