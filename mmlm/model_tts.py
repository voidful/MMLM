import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import Optional, Union, List
import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from mmlm.common import initialize_language_model, FocalLoss
from mmlm.utility import align_and_sum_embeddings, align_logits_and_labels, prepare_labels, \
    initialize_head_weight_from_lm


class MMLMTTSConfig(PretrainedConfig):
    model_type = "mmlmtts"

    def __init__(self, lm_model_name="voidful/SmolLM2-360M-Instruct-Whisper",
                 num_heads=8, codebook_size=2048,
                 speaker_emb_dim=192, **kwargs):
        super().__init__(**kwargs)
        self.lm_model_name = lm_model_name
        self.num_heads = num_heads
        self.codebook_size = codebook_size
        self.speaker_emb_dim = speaker_emb_dim


class MMLMTTS(PreTrainedModel):
    config_class = MMLMTTSConfig

    def __init__(self, config: MMLMTTSConfig):
        super().__init__(config)
        self.config = config
        self.num_heads = config.num_heads
        self._initialize_model(config)

    def _initialize_model(self, config: MMLMTTSConfig):
        initialize_language_model(self, config)
        self._initialize_tts_components()

    def _initialize_tts_components(self):
        lm_embeddings = self.lm_model.get_input_embeddings().weight
        vocab_size, embedding_dim = lm_embeddings.size()

        self.codec_decoding_head = nn.ModuleList([
            (initialize_head_weight_from_lm(lm_embeddings, vocab_size, embedding_dim, self.config.codebook_size)
             if head_num > 0 else
             initialize_head_weight_from_lm(lm_embeddings, vocab_size, embedding_dim, self.config.codebook_size + 3))
            for head_num in range(self.num_heads)
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

    def embed_audio(self, input_values):
        weighted_embeds = torch.stack([
            head(input_values[i, :]) * F.softmax(self.learned_layer_weight[i], dim=0)
            for i, head in enumerate(self.codec_decoding_head)
        ], dim=0).sum(dim=0)
        return weighted_embeds

    def embed_system_prompt(self, synthesis_text_ids, speaker_emb):
        # Prepare the template and tokenize components
        template = self.tokenizer.apply_chat_template(
            [{"role": "system",
              "content": "Synth [SYN_TEXT] with reference speech <begin_of_speech>[REFSPEECH]<end_of_speech>."}],
            tokenize=False,
            add_generation_prompt=False,
        )
        parts = template.split("[REFSPEECH]")[0].split("[SYN_TEXT]") + [template.split("[REFSPEECH]")[1]]
        tokens = [self.tokenizer.encode(part, add_special_tokens=False, return_tensors='pt').squeeze().to(
            self.lm_model.device) for part in
            parts]

        # Generate embeddings
        speaker_emb = self.speaker_adapter(speaker_emb)
        synthesis_text_emb = self.embed_text(synthesis_text_ids)[0]
        # Concatenate embeddings and return
        return torch.cat(
            [self.embed_text(tokens[0]), synthesis_text_emb, self.embed_text(tokens[1]), speaker_emb,
             self.embed_text(tokens[2])]
        ).unsqueeze(0)

    def forward(
            self,
            input_values: Optional[List[torch.LongTensor]] = None,
            speaker_emb: Optional[torch.LongTensor] = None,
            tts_text: Optional[torch.LongTensor] = None,
            tts_text_with_pad: Optional[torch.LongTensor] = None,
            tts_label_with_pad: Optional[torch.LongTensor] = None,
            codec_label: Optional[List[torch.LongTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[torch.Tensor, CausalLMOutputWithPast]:
        tts_text_with_pad, tts_label_with_pad = prepare_labels(self.tokenizer, tts_text_with_pad, tts_label_with_pad)
        mimi_embeds = self.embed_audio(input_values)
        text_embeds = self.embed_text(tts_text_with_pad)
        inputs_embeds = align_and_sum_embeddings(mimi_embeds, text_embeds)
        system_embeds = self.embed_system_prompt(tts_text, speaker_emb)
        if codec_label is not None:
            codec_label = F.pad(codec_label, (system_embeds.shape[1], 0), value=-100)
        if tts_label_with_pad is not None:
            tts_label_with_pad = F.pad(tts_label_with_pad, (system_embeds.shape[1], 0), value=-100)
        outputs = self.lm_model.model(
            inputs_embeds=torch.cat([system_embeds, inputs_embeds], dim=1).to(self.lm_model.dtype),
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        total_loss = 0.0
        decoded_logits = self.lm_head(outputs.last_hidden_state)
        logits, labels = align_logits_and_labels(decoded_logits, tts_label_with_pad)
        loss_fct = FocalLoss()
        total_loss += loss_fct(decoded_logits.view(-1, logits.size(-1)), labels.view(-1))
        for i, decoding_head in enumerate(self.tts_decoding_head):
            if codec_label is not None and codec_label[i] is not None:
                head_logits = decoding_head(outputs.last_hidden_state)
                logits, labels = align_logits_and_labels(head_logits, codec_label[i,:].unsqueeze(0))
                total_loss += loss_fct(head_logits.view(-1, logits.size(-1)), labels.view(-1))

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

    def stream_generate(self, speaker_emb, tts_text, max_length=50):
        text_sequence = [self.tokenizer.bos_token_id]
        audio_sequences = {i: [0] for i in range(0, len(self.tts_decoding_head))}
        tts_text = self.tokenizer(tts_text, return_tensors='pt')['input_ids'].to("cuda")
        self.eval()
        with torch.no_grad():
            for step in range(max_length):
                input_values = torch.tensor([[audio_sequences[i]] for i in audio_sequences]).to(torch.long).to("cuda")
                tts_text_with_pad = torch.tensor([text_sequence]).to(torch.long).to("cuda")
                outputs = self.forward(input_values=input_values, speaker_emb=speaker_emb,
                                       tts_text=tts_text,
                                       tts_text_with_pad=tts_text_with_pad,
                                       tts_label_with_pad=tts_text_with_pad)
                next_token = torch.argmax(self.lm_head(outputs.logits)[:, -1, :], dim=-1).item()
                text_sequence.append(next_token)
                for head_idx in audio_sequences:
                    head_logits = self.tts_decoding_head[head_idx](outputs.logits.to(torch.float))
                    next_audio_unit = torch.argmax(head_logits[:, -1, :], dim=-1).item()
                    audio_sequences[head_idx].append(next_audio_unit)
                if next_token == self.tokenizer.bos_token_id or audio_sequences[0][-1] > 2048:
                    break

        audio_sequences = [audio_sequences[i] for i in range(0, len(audio_sequences))]
        real_audio = [[audio_sequences[0][:-1]] + [seq[1:] for seq in audio_sequences[1:]]]
        return real_audio,self.tokenizer.decode(text_sequence)
