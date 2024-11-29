import os

from mmlm.listener import ListenFeatureExtractor

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import Optional, Union, List
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from mmlm.common import initialize_language_model
from mmlm.utility import align_and_sum_embeddings, align_logits_and_labels, prepare_labels, \
    initialize_head_weight_from_lm


class MMLMConfig(PretrainedConfig):
    model_type = "mmlm"

    def __init__(self, lm_model_name="voidful/SmolLM2-360M-Instruct-Whisper",
                 num_heads=8, codebook_size=2048,
                 speaker_emb_dim=192, **kwargs):
        super().__init__(**kwargs)
        self.lm_model_name = lm_model_name
        self.num_heads = num_heads
        self.codebook_size = codebook_size + 3
        self.speaker_emb_dim = speaker_emb_dim


class MMLM(PreTrainedModel):
    config_class = MMLMConfig

    def __init__(self, config: MMLMConfig):
        super().__init__(config)
        self.config = config
        self.num_heads = config.num_heads
        self._initialize_model(config)

    def _initialize_model(self, config: MMLMConfig):
        initialize_language_model(self, config)
        self._initialize_tts_components()
        self._initialize_asr_components()
        self.asr_head = self.lm_head
        self.asr_head.weight = torch.nn.Parameter(self.lm_head.weight.clone())
        self.tts_head = self.lm_head
        self.tts_head.weight = torch.nn.Parameter(self.lm_head.weight.clone())

    def _initialize_asr_components(self):
        self.listener = ListenFeatureExtractor()
        embedding_dim = self.lm_model.get_input_embeddings().weight.size(-1)
        self.adapter = nn.Linear(512, embedding_dim)

    def _initialize_tts_components(self):
        lm_embeddings = self.lm_model.get_input_embeddings().weight
        vocab_size, embedding_dim = lm_embeddings.size()

        self.codec_decoding_head = nn.ModuleList([
            initialize_head_weight_from_lm(lm_embeddings, vocab_size, embedding_dim, self.config.codebook_size)
            for _ in range(self.num_heads)
        ])
        self.tts_decoding_head = nn.ModuleList([
            nn.Linear(embedding_dim, self.config.codebook_size, bias=False) for _ in range(self.num_heads)
        ])

        self.speaker_adapter = nn.Linear(self.config.speaker_emb_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.learned_layer_weight = nn.Parameter(
            torch.arange(self.num_heads, 0, step=-1).float().view(self.num_heads, 1, 1, 1)
        )

    def embed_text(self, input_ids):
        return self.lm_model.get_input_embeddings()(input_ids)

    def embed_asr_audio(self, input_values: torch.Tensor) -> torch.Tensor:
        features = self.listener(input_values)
        adapted_features = self.adapter(features.permute(0, 2, 1))
        return adapted_features

    def embed_tts_audio(self, input_values):
        weighted_embeds = torch.stack([
            head(input_values[i, :]) * F.softmax(self.learned_layer_weight[i], dim=0)
            for i, head in enumerate(self.codec_decoding_head)
        ], dim=0).sum(dim=0)
        return weighted_embeds

    def embed_system_prompt(self, synthesis_text_ids, speaker_emb):
        # Prepare the template and tokenize components
        template = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": "Synth [SYN_TEXT] with reference speech [REFSPEECH]"}],
            tokenize=False,
            add_generation_prompt=False,
        )
        parts = template.split("[REFSPEECH]")[0].split("[SYN_TEXT]") + [template.split("[REFSPEECH]")[1]]
        tokens = [self.tokenizer.encode(part, add_special_tokens=False, return_tensors='pt').squeeze().to(
            self.lm_model.device) for part in
            parts]

        # Generate embeddings
        speaker_emb = self.speaker_adapter(speaker_emb)
        synthesis_text_emb = self.embed_text(synthesis_text_ids).squeeze()

        # Concatenate embeddings and return
        return torch.cat(
            [self.embed_text(tokens[0]), synthesis_text_emb, self.embed_text(tokens[1]), speaker_emb,
             self.embed_text(tokens[2])]
        ).unsqueeze(0)

    def forward(
            self,
            input_values: Optional[torch.FloatTensor] = None,
            asr_texts: Optional[torch.LongTensor] = None,
            asr_labels: Optional[torch.LongTensor] = None,
            codec_input: Optional[List[torch.LongTensor]] = None,
            speaker_emb: Optional[torch.LongTensor] = None,
            tts_text: Optional[torch.LongTensor] = None,
            tts_label: Optional[torch.LongTensor] = None,
            codec_label: Optional[List[torch.LongTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[torch.Tensor, CausalLMOutputWithPast]:
        tts_text, tts_label = prepare_labels(self.tokenizer, tts_text, tts_label)
        asr_embeds = self.embed_asr_audio(input_values)
        mimi_embeds = self.embed_tts_audio(codec_input)
        text_embeds = self.embed_text(tts_text)

        inputs_embeds = align_and_sum_embeddings(mimi_embeds, text_embeds)
        inputs_embeds = align_and_sum_embeddings(inputs_embeds, asr_embeds)

        system_embeds = self.embed_system_prompt(tts_text, speaker_emb)
        tts_label = F.pad(tts_label, (system_embeds.shape[1], 0), value=-100)
        asr_labels = F.pad(asr_labels, (system_embeds.shape[1], 0), value=-100)

        outputs = self.lm_model.model(
            inputs_embeds=torch.cat([system_embeds, inputs_embeds], dim=1),
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return self._compute_loss_and_output(outputs, asr_labels, tts_label, codec_label)

    def _compute_loss_and_output(self, outputs, asr_labels, tts_label, codec_label):
        total_loss = 0.0
        loss_fct = CrossEntropyLoss()

        decoded_logits = self.tts_head(outputs.last_hidden_state)
        logits, labels = align_logits_and_labels(decoded_logits, tts_label)
        total_loss += loss_fct(decoded_logits.view(-1, logits.size(-1)), labels.view(-1))

        decoded_logits = self.asr_head(outputs.last_hidden_state)
        logits, labels = align_logits_and_labels(decoded_logits, asr_labels)
        total_loss += loss_fct(decoded_logits.view(-1, logits.size(-1)), labels.view(-1))

        for i, decoding_head in enumerate(self.tts_decoding_head):
            if codec_label and codec_label[i] is not None:
                head_logits = decoding_head(outputs.last_hidden_state)
                logits, labels = align_logits_and_labels(head_logits, codec_label[i])
                total_loss += loss_fct(head_logits.view(-1, logits.size(-1)), labels.view(-1))

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=decoded_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.lm_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.lm_model.gradient_checkpointing_disable()
