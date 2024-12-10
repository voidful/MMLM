import os
from collections import deque
from typing import Optional, Union, List

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from mmlm.listener import ListenFeatureExtractor
from mmlm.common import initialize_language_model, FocalLoss
from mmlm.utility import (
    align_and_sum_embeddings,
    align_logits_and_labels,
    prepare_labels,
    initialize_head_weight_from_lm,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MMLMConfig(PretrainedConfig):
    model_type = "mmlm"

    def __init__(
            self,
            lm_model_name="voidful/SmolLM2-360M-Instruct-Whisper",
            num_heads=8,
            codebook_size=2048,
            speaker_emb_dim=192,
            step_duration=0.08,
            sampling_rate=24000,
            queue_duration=3600,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.lm_model_name = lm_model_name
        self.num_heads = num_heads
        self.codebook_size = codebook_size
        self.speaker_emb_dim = speaker_emb_dim
        self.step_duration = step_duration
        self.sampling_rate = sampling_rate
        self.step_size = int(sampling_rate * step_duration)
        self.queue_duration = queue_duration
        self.queue_length = int(queue_duration * sampling_rate)


class MMLM(PreTrainedModel):
    config_class = MMLMConfig

    def __init__(self, config: MMLMConfig):
        super().__init__(config)
        self.config = config
        self.num_heads = config.num_heads

        self._initialize_model(config)
        self._initialize_queues()

    def _initialize_model(self, config: MMLMConfig):
        initialize_language_model(self, config)
        self._initialize_tts_components()
        self._initialize_asr_components()

        # Initialize ASR and TTS heads with cloned weights from the language model head
        self.asr_head = nn.Linear(
            self.lm_head.in_features, self.lm_head.out_features, bias=False
        )
        self.asr_head.weight = nn.Parameter(self.lm_head.weight.clone())

        self.tts_head = nn.Linear(
            self.lm_head.in_features, self.lm_head.out_features, bias=False
        )
        self.tts_head.weight = nn.Parameter(self.lm_head.weight.clone())

    def _initialize_asr_components(self):
        self.listener = ListenFeatureExtractor()
        embedding_dim = self.lm_model.get_input_embeddings().embedding_dim
        self.adapter = nn.Linear(512, embedding_dim)

    def _initialize_tts_components(self):
        lm_embeddings = self.lm_model.get_input_embeddings().weight
        vocab_size, embedding_dim = lm_embeddings.size()

        self.codec_decoding_heads = nn.ModuleList([
            initialize_head_weight_from_lm(
                lm_embeddings, vocab_size, embedding_dim, self.config.codebook_size
            )
            for _ in range(self.num_heads)
        ])

        self.tts_decoding_heads = nn.ModuleList([
            nn.Linear(embedding_dim, self.config.codebook_size, bias=False)
            for _ in range(self.num_heads)
        ])

        self.speaker_adapter = nn.Linear(self.config.speaker_emb_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.learned_layer_weight = nn.Parameter(
            torch.linspace(self.num_heads, 1, steps=self.num_heads).view(self.num_heads, 1, 1, 1)
        )

    def _initialize_queues(self):
        self.listener_queue = deque(maxlen=self.config.queue_length)
        self.listener_text_queue = deque([self.tokenizer.bos_token_id], maxlen=self.config.queue_length)
        self.speaker_text_queue = deque([self.tokenizer.bos_token_id], maxlen=self.config.queue_length)

        self.speaker_queues = {
            i: deque([0], maxlen=self.config.queue_length)
            for i in range(self.num_heads)
        }

    def embed_asr_text(self, input_ids):
        return self.asr_head.weight[input_ids]

    def embed_tts_text(self, input_ids):
        return self.tts_head.weight[input_ids]

    def embed_text(self, input_ids):
        return self.lm_model.get_input_embeddings()(input_ids)

    def embed_asr_audio(self, input_values: torch.Tensor) -> torch.Tensor:
        features = self.listener(input_values)
        adapted_features = self.adapter(features.permute(0, 2, 1))
        return adapted_features

    def embed_tts_audio(self, input_values):
        weighted_embeds = torch.stack([
            head(input_values[i]) * F.softmax(self.learned_layer_weight[i], dim=0)
            for i, head in enumerate(self.codec_decoding_heads)
        ], dim=0).sum(dim=0)
        return weighted_embeds

    def embed_system_prompt(self, speaker_emb):
        template = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": "Chat with reference speech <ref_speech_start>[REFSPEECH]<ref_speech_end>"}],
            tokenize=False,
            add_generation_prompt=False,
        )
        parts = template.split("[REFSPEECH]")
        tokens = [
            self.tokenizer.encode(part, add_special_tokens=False, return_tensors='pt').squeeze().to(
                self.lm_model.device
            )
            for part in parts
        ]
        speaker_emb = self.speaker_adapter(speaker_emb)
        return torch.cat([self.embed_text(tokens[0]), speaker_emb, self.embed_text(tokens[1])]).unsqueeze(0)

    def forward(
            self,
            input_values: Optional[torch.FloatTensor] = None,
            listener_texts: Optional[torch.LongTensor] = None,
            listener_text_labels: Optional[torch.LongTensor] = None,
            speaker_codecs: Optional[List[torch.LongTensor]] = None,
            speaker_codec_labels: Optional[List[torch.LongTensor]] = None,
            speaker_embs: Optional[torch.LongTensor] = None,
            speaker_texts: Optional[torch.LongTensor] = None,
            speaker_text_labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[torch.Tensor, CausalLMOutputWithPast]:

        listener_texts, listener_text_labels = prepare_labels(
            self.tokenizer, listener_texts, listener_text_labels
        )
        speaker_texts, speaker_text_labels = prepare_labels(
            self.tokenizer, speaker_texts, speaker_text_labels
        )

        listener_embeds = self.embed_asr_audio(input_values)
        speaker_embeds = self.embed_tts_audio(speaker_codecs)

        listener_text_embeds = self.embed_asr_text(listener_texts)
        speaker_text_embeds = self.embed_tts_text(speaker_texts)

        listener_embeds = align_and_sum_embeddings(listener_embeds, listener_text_embeds)
        speaker_embeds = align_and_sum_embeddings(speaker_embeds, speaker_text_embeds)
        input_embeds = align_and_sum_embeddings(listener_embeds, speaker_embeds)
        system_embeds = self.embed_system_prompt(speaker_embs)

        if speaker_codec_labels is not None:
            padding = (system_embeds.shape[1], 0)
            speaker_codec_labels = F.pad(speaker_codec_labels, padding, value=-100)
        if speaker_text_labels is not None:
            padding = (system_embeds.shape[1], 0)
            speaker_text_labels = F.pad(speaker_text_labels, padding, value=-100)
        if listener_text_labels is not None:
            padding = (system_embeds.shape[1], 0)
            listener_text_labels = F.pad(listener_text_labels, padding, value=-100)

        outputs = self.lm_model.model(
            inputs_embeds=torch.cat([system_embeds, input_embeds], dim=1).to(self.lm_model.dtype),
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return self._compute_loss_and_output(
            outputs, listener_text_labels, speaker_text_labels, speaker_codec_labels
        )

    def _compute_loss_and_output(
            self, outputs, listener_text_labels, speaker_text_label, speaker_label
    ):
        total_loss = 0.0
        loss_fct = FocalLoss()

        decoded_logits = self.tts_head(outputs.last_hidden_state)
        logits, labels = align_logits_and_labels(decoded_logits, speaker_text_label)
        tts_loss = loss_fct(decoded_logits.view(-1, logits.size(-1)), labels.view(-1))
        total_loss += loss_fct(decoded_logits.view(-1, logits.size(-1)), labels.view(-1))
        decoded_logits = self.asr_head(outputs.last_hidden_state)
        logits, labels = align_logits_and_labels(decoded_logits, listener_text_labels)
        asr_loss = loss_fct(decoded_logits.view(-1, logits.size(-1)), labels.view(-1))
        total_loss += asr_loss
        for i, decoding_head in enumerate(self.tts_decoding_heads):
            if speaker_label is not None and speaker_label[i] is not None:
                head_logits = decoding_head(outputs.last_hidden_state)
                logits, labels = align_logits_and_labels(head_logits, speaker_label[i].unsqueeze(0))
                layer_loss = loss_fct(head_logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += layer_loss

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.lm_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.lm_model.gradient_checkpointing_disable()

    def stream_generate(
            self, speaker_emb, listener_values, listener_values_rest, listener_text=None, max_length=50
    ):
        self.eval()
        self._initialize_queues()
        with torch.no_grad():
            for start in range(0, len(listener_values), self.config.step_size):
                chunk = listener_values[start: start + self.config.step_size]
                self.listener_queue.extend(chunk)
                current_audio = torch.tensor(list(self.listener_queue)).unsqueeze(0).to("cuda")
                listener_text_tensor = torch.tensor(list(self.listener_text_queue)).unsqueeze(0).to("cuda")
                speaker_queues_tensor = torch.tensor([
                    list(queue) for queue in self.speaker_queues.values()
                ]).to("cuda").to(torch.long)
                speaker_text_tensor = torch.tensor(list(self.speaker_text_queue)).unsqueeze(0).to("cuda")
                outputs = self.forward(
                    current_audio,
                    listener_texts=listener_text_tensor,
                    listener_text_labels=listener_text_tensor,
                    speaker_codecs=speaker_queues_tensor,
                    speaker_embs=speaker_emb,
                    speaker_texts=speaker_text_tensor,
                    speaker_text_labels=speaker_text_tensor,
                )

                asr_logits = self.asr_head(outputs.logits)
                next_asr_token = torch.argmax(asr_logits, dim=-1)[:, -1].item()
                self.listener_text_queue.append(next_asr_token)

                tts_logits = self.tts_head(outputs.logits)
                next_tts_token = torch.argmax(tts_logits, dim=-1)[:, -1].item()
                self.speaker_text_queue.append(next_tts_token)

                for idx, decoding_head in enumerate(self.tts_decoding_heads):
                    head_logits = decoding_head(outputs.logits.to(torch.float))
                    next_audio_unit = torch.argmax(head_logits[:, -1, :], dim=-1).item()
                    self.speaker_queues[idx].append(next_audio_unit)

        decode_start_pos = len(self.listener_text_queue)
        for _ in range(max_length):
            current_audio = torch.tensor(list(self.listener_queue)).unsqueeze(0).to("cuda")
            listener_text_tensor = torch.tensor(list(self.listener_text_queue)).unsqueeze(0).to("cuda")
            speaker_queues_tensor = torch.tensor([
                list(queue) for queue in self.speaker_queues.values()
            ]).to(torch.long).to("cuda")
            speaker_text_tensor = torch.tensor(list(self.speaker_text_queue)).unsqueeze(0).to("cuda")
            outputs = self.forward(
                current_audio,
                listener_texts=listener_text_tensor,
                listener_text_labels=listener_text_tensor,
                speaker_codecs=speaker_queues_tensor,
                speaker_embs=speaker_emb,
                speaker_texts=speaker_text_tensor,
                speaker_text_labels=speaker_text_tensor,
            )

            tts_logits = self.tts_head(outputs.logits)
            next_tts_token = torch.argmax(tts_logits, dim=-1)[:, -1].item()

            if (
                    next_tts_token == self.tokenizer.bos_token_id
                    or any(q[-1] > self.config.codebook_size for q in self.speaker_queues.values())
                    or len(self.speaker_text_queue) > max_length
            ):
                break

            if _ == 0:
                self.speaker_text_queue.append(self.tokenizer.encode("[END_PAD]", add_special_tokens=False)[0])
            else:
                self.speaker_text_queue.append(next_tts_token)

            self.listener_text_queue.append(self.tokenizer.encode("[PAD]", add_special_tokens=False)[0])
            self.listener_queue.extend(listener_values_rest[_ * self.config.step_size: (_ + 1) * self.config.step_size])

            for idx, decoding_head in enumerate(self.tts_decoding_heads):
                head_logits = decoding_head(outputs.logits.to(torch.float))
                next_audio_unit = torch.argmax(head_logits[:, -1, :], dim=-1).item()
                self.speaker_queues[idx].append(next_audio_unit)

        audio_sequences = [
            list(queue)[decode_start_pos:] for queue in self.speaker_queues.values()
        ]
        mimi_code = [seq[:-1] if idx == 0 else seq[1:] for idx, seq in enumerate(audio_sequences)]
        generated_text = self.tokenizer.decode(
            list(self.speaker_text_queue)[decode_start_pos:]
        )
        return mimi_code, generated_text
