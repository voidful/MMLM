import os
from collections import deque
from typing import Optional, Union, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from mmlm.common import initialize_language_model, FocalLoss
from mmlm.listener import ListenFeatureExtractor
from mmlm.utility import align_and_sum_embeddings, prepare_labels

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MMLMASRConfig(PretrainedConfig):
    model_type = "mmlmasr"

    def __init__(
        self,
        lm_model_name="voidful/SmolLM2-360M-Instruct-Whisper",
        step_duration=0.08,
        sampling_rate=24000,
        queue_duration=3600,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lm_model_name = lm_model_name
        self.step_duration = step_duration
        self.sampling_rate = sampling_rate
        self.step_size = int(sampling_rate * step_duration)
        self.queue_duration = queue_duration
        self.queue_length = int(queue_duration * sampling_rate)


class MMLMASR(PreTrainedModel):
    config_class = MMLMASRConfig

    def __init__(self, config: MMLMASRConfig):
        super().__init__(config)
        self.config = config
        initialize_language_model(self, config)
        self._initialize_asr_components()
        self.audio_queue = deque(maxlen=self.config.queue_length)
        self.text_queue = deque([self.tokenizer.bos_token_id], maxlen=self.config.queue_length)

    def _initialize_asr_components(self):
        self.listener = ListenFeatureExtractor()
        embedding_dim = self.lm_model.get_input_embeddings().embedding_dim
        self.adapter = nn.Linear(512, embedding_dim)

    def embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedder = self.lm_model.get_input_embeddings()
        return embedder(input_ids)

    def embed_audio(self, input_values: torch.Tensor) -> torch.Tensor:
        features = self.listener(input_values)
        adapted_features = self.adapter(features.permute(0, 2, 1))
        return adapted_features

    def embed_system_prompt(
        self,
        prompt_text: str = "you are a helpful ASR model, please help me to understand the audio.",
    ) -> torch.Tensor:
        template = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": prompt_text}],
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        system_embeds = self.embed_text(template.to(self.lm_model.device))
        return system_embeds

    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        asr_texts: Optional[torch.LongTensor] = None,
        asr_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        asr_texts, asr_labels = prepare_labels(self.tokenizer, asr_texts, asr_labels)

        # Encode inputs
        audio_embeds = self.embed_audio(input_values)
        text_embeds = self.embed_text(asr_texts)
        inputs_embeds = align_and_sum_embeddings(audio_embeds, text_embeds)
        system_embeds = self.embed_system_prompt()

        outputs = self.lm_model(
            inputs_embeds=torch.cat([system_embeds, inputs_embeds], dim=1).to(self.lm_model.dtype),
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

    def stream_generate(self, audio_seq, max_length: int = 50) -> str:
        self.eval()
        with torch.no_grad():
            for start in range(0, len(audio_seq), self.config.step_size):
                # Extract the current audio chunk
                chunk = audio_seq[start : start + self.config.step_size]
                self.audio_queue.extend(chunk)  # Add chunk to the queue

                # Convert the audio queue to a tensor
                current_audio = torch.tensor(list(self.audio_queue)).unsqueeze(0).to(self.lm_model.device)
                current_text = torch.tensor(list(self.text_queue)).unsqueeze(0).to(self.lm_model.device)

                outputs = self.forward(
                    input_values=current_audio,
                    asr_texts=current_text,
                    asr_labels=current_text,
                )

                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).item()

                # Append the decoded token to the sequence
                self.text_queue.append(next_token)

                # Stop decoding if EOS token is generated
                if next_token == self.tokenizer.eos_token_id or len(self.text_queue) >= max_length:
                    break

        decoded_text = self.tokenizer.decode(
            list(self.text_queue),
            skip_special_tokens=True,
        )
        return decoded_text

    def clear_stream(self):
        self.audio_queue.clear()
        self.text_queue.clear()
        self.text_queue.append(self.tokenizer.bos_token_id)
