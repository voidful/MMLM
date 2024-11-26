from typing import Optional, Union, Tuple, List
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.configuration_utils import PretrainedConfig
import torch
import torch.nn.functional as F
from mmlm.synth import SynthFeatureExtractor

from transformers import PretrainedConfig

from mmlm.utility import add_bos_eos_tokens_if_not_exist, align_and_sum_embeddings, align_logits_and_labels


class MMLMTTSConfig(PretrainedConfig):
    model_type = "mmlmtts"

    def __init__(self, lm_model_name="voidful/SmolLM2-360M-Instruct-TTS", num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.lm_model_name = lm_model_name
        self.num_heads = num_heads
        # Add any additional configuration parameters here


import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# Assuming AutoLigerKernelForCausalLM is similar to AutoModelForCausalLM
# Replace it with AutoModelForCausalLM if AutoLigerKernelForCausalLM is unavailable
from transformers import AutoModelForCausalLM


class MMLMTTS(PreTrainedModel):
    config_class = MMLMTTSConfig

    def __init__(self, config: MMLMTTSConfig):
        super().__init__(config)
        self.config = config
        self.num_heads = config.num_heads

        self._initialize_language_model(config)
        self._initialize_custom_components()

    def _initialize_language_model(self, config: MMLMTTSConfig):
        self.lm_model = AutoModelForCausalLM.from_pretrained(
            config.lm_model_name,
            # Add additional parameters as needed
        )
        self.lm_head = self.lm_model.lm_head
        self.tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _initialize_custom_components(self):
        # Retrieve the language model's input embeddings
        lm_embeddings = self.lm_model.get_input_embeddings().weight  # Shape: (vocab_size, embedding_dim)
        vocab_size, embedding_dim = lm_embeddings.size()

        # Initialize multi_decoding_head with embeddings sampled from LM's embeddings
        self.multi_decoding_head = nn.ModuleList([
            nn.Embedding(2048, embedding_dim) for _ in range(self.num_heads)
        ])

        # Initialize linear decoding heads
        self.linear_decoding_head = nn.ModuleList([
            nn.Linear(embedding_dim, 2048, bias=False) for _ in range(self.num_heads)
        ])

        for i, head in enumerate(self.multi_decoding_head):
            # Randomly sample 2048 indices from LM's embeddings
            sampled_indices = torch.randint(0, vocab_size, (2048,))
            sampled_weights = lm_embeddings[sampled_indices].clone().detach()
            head.weight = nn.Parameter(sampled_weights)

            # Tie weights between embedding and linear layers
            self.linear_decoding_head[i].weight = nn.Parameter(sampled_weights)

        # Speaker adapter to align speaker embeddings with LM embeddings
        self.speaker_adapter = nn.Linear(192, embedding_dim)

        # Layer normalization (if needed)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Learnable weights for each decoding head
        self.learned_layer_weight = nn.Parameter(
            torch.arange(self.num_heads, 0, step=-1).float().view(self.num_heads, 1, 1, 1)
        )

        self._freeze_lm_parameters()

    def _freeze_lm_parameters(self):
        for param in self.lm_model.parameters():
            param.requires_grad = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = MMLMTTSConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)

    def _tokenize_and_encode(self, text: str) -> torch.Tensor:
        return self.tokenizer.encode(text, add_special_tokens=False, return_tensors='pt').squeeze()

    def encode_system_prompt(self, synthesis_text_ids, speaker_emb):
        template = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": f"Synth [SYN_TEXT] with reference speech [REFSPEECH]"}],
            tokenize=False,
            add_generation_prompt=False,
        )
        t0, t1 = template.split("[REFSPEECH]")
        t01, t02 = t0.split("[SYN_TEXT]")
        t01 = self._tokenize_and_encode(t01).to(self.lm_model.device)
        t02 = self._tokenize_and_encode(t02).to(self.lm_model.device)
        t1 = self._tokenize_and_encode(t1).to(self.lm_model.device)
        speaker_emb = speaker_emb.to(self.lm_model.device)
        return torch.cat([
            self.encode_text(t01),
            self.encode_text(synthesis_text_ids).squeeze(),
            self.encode_text(t02),
            self.speaker_adapter(speaker_emb),
            self.encode_text(t1)
        ]).unsqueeze(0)

    def encode_text(self, input_ids):
        return self.lm_model.get_input_embeddings()(input_ids)

    def _process_audio_embeddings(self, input_values):
        mimi_embeds = []
        for i, head in enumerate(self.multi_decoding_head):
            embed = head(input_values[:, i, :])
            mimi_embeds.append(embed)
        # Stack embeddings: (num_heads, batch_size, embedding_dim)
        mimi_embeds = torch.stack(mimi_embeds, dim=0)
        # Apply softmax to learned_layer_weight and weight the embeddings
        weights = F.softmax(self.learned_layer_weight, dim=0)  # (num_heads, 1, 1)
        weighted_embeds = mimi_embeds * weights  # Broadcasting
        # Sum over heads: (batch_size, embedding_dim)
        mimi_embeds = weighted_embeds.sum(dim=0)
        return mimi_embeds

    def forward(
            self,
            input_values: Optional[List[torch.LongTensor]] = None,  # Shape: (batch_size, num_heads, feature_dim)
            speaker_emb: Optional[torch.LongTensor] = None,  # Shape: (batch_size,)
            tts_text: Optional[torch.LongTensor] = None,  # List of labels per head
            text_label: Optional[torch.LongTensor] = None,  # List of labels per head
            codec_label: Optional[List[torch.LongTensor]] = None,  # List of labels per head
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[torch.Tensor, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if tts_text is None and text_label is None:
            assert "Either tts_text or text_label must be provided."

        if text_label is not None:
            text_label = add_bos_eos_tokens_if_not_exist(self.tokenizer, text_label)
        if tts_text is not None:
            tts_text = add_bos_eos_tokens_if_not_exist(self.tokenizer, tts_text)

        if tts_text is None:
            # text_label shifted by 1, return a tensor of shape (batch_size, seq_len) in one line
            tts_text = text_label[:, :-1].contiguous()
            text_label = text_label[:, 1:].contiguous()
        elif text_label is None:
            # tts_text shifted by 1, return a tensor of shape (batch_size, seq_len) in one line
            text_label = tts_text[:, 1:].contiguous()
            tts_text = tts_text[:, :-1].contiguous()

        # Process embeddings
        mimi_embeds = self._process_audio_embeddings(input_values)  # (batch_size, embedding_dim)
        text_embeds = self.encode_text(tts_text)  # Assuming first head's labels correspond to text

        # Align and sum embeddings
        inputs_embeds = align_and_sum_embeddings(mimi_embeds, text_embeds)
        system_embeds = self.encode_system_prompt(tts_text, speaker_emb)
        # Calculate the length of system_embeds
        system_embed_length = system_embeds.shape[1]
        # Adjust text_label by prepending -100 to mask out system_embeds
        text_label = F.pad(text_label, (system_embed_length, 0), value=-100)

        inputs_embeds = torch.cat([system_embeds, inputs_embeds],
                                  dim=1)
        # Pass through the language model
        outputs = self.lm_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Retrieve the last hidden state
        if output_hidden_states:
            last_hidden_state = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)
        else:
            last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # Initialize loss and logits
        total_loss = 0.0
        decoded_logits = self.lm_head(last_hidden_state)
        logits, label = align_logits_and_labels(decoded_logits, text_label)
        loss_fct = CrossEntropyLoss()
        lm_loss = loss_fct(decoded_logits.view(-1, logits.size(-1)), label.view(-1))
        total_loss += lm_loss
        # Iterate over each decoding head
        for i, decoding_head in enumerate(self.linear_decoding_head):
            # Compute loss if labels are provided
            if codec_label is not None and codec_label[i] is not None:
                # Pass the last hidden state through the decoding head
                decoded_logits = decoding_head(last_hidden_state)  # (batch_size, seq_len, vocab_size)
                # Align logits and labels
                logits, label = align_logits_and_labels(decoded_logits, codec_label[i])
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(decoded_logits.view(-1, logits.size(-1)), label.view(-1))
                total_loss += loss

        if return_dict:
            return CausalLMOutputWithPast(
                loss=total_loss if text_label is not None else None,
                logits=decoded_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            return (total_loss, decoded_logits) if text_label is not None else decoded_logits

    def _align_logits_and_labels(self, logits, labels):
        logits_len = logits.size(1)
        labels_len = labels.size(1)
        if labels_len < logits_len:
            labels = F.pad(labels, (0, logits_len - labels_len), value=-100)
        elif labels_len > logits_len:
            labels = labels[:, :logits_len]
        return logits, labels

    def generate(self, input_ids, audio_feature=None, max_length=50):
        self.eval()
        generated = input_ids
        begin_gen_pos = input_ids.size(1)

        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(input_values=audio_feature, labels=[generated])
                next_token = torch.argmax(outputs.logits[0][:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        return generated[:, begin_gen_pos:-1]
