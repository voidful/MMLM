import copy
from typing import Optional, Union, Tuple, List
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.configuration_utils import PretrainedConfig
import torch
import torch.nn.functional as F

from mmlm.listener import ListenFeatureExtractor
from mmlm.synth import SynthFeatureExtractor

from transformers import PretrainedConfig

from mmlm.utility import add_bos_eos_tokens_if_not_exist, align_and_sum_embeddings, align_logits_and_labels
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# Assuming AutoLigerKernelForCausalLM is similar to AutoModelForCausalLM
# Replace it with AutoModelForCausalLM if AutoLigerKernelForCausalLM is unavailable
from liger_kernel.transformers import AutoLigerKernelForCausalLM


class MMLMConfig(PretrainedConfig):
    model_type = "mmlm"

    def __init__(self, lm_model_name="voidful/SmolLM2-360M-Instruct", num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.lm_model_name = lm_model_name
        self.num_heads = num_heads
        # Add any additional configuration parameters here


class MMLM(PreTrainedModel):
    config_class = MMLMConfig

    def __init__(self, config: MMLMConfig):
        super().__init__(config)
        self.config = config
        self.num_heads = config.num_heads

        self._initialize_language_model(config)
        self._initialize_custom_components()

    def _initialize_language_model(self, config: MMLMConfig):
        self.lm_model = AutoLigerKernelForCausalLM.from_pretrained(
            config.lm_model_name,
            trust_remote_code=True,
            use_cache=False,
            torch_dtype=torch.bfloat16,
            rope=True,
            swiglu=True,
            cross_entropy=True,
            fused_linear_cross_entropy=False,
            rms_norm=True,
        )
        self.listener_head = copy.deepcopy(self.lm_model.lm_head)
        self.speaker_head = copy.deepcopy(self.lm_model.lm_head)
        self.tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _initialize_custom_components(self):
        self.listener = ListenFeatureExtractor()
        self.listener_adapter = nn.Linear(
            512, self.lm_model.get_input_embeddings().weight.shape[-1]
        )

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
            torch.arange(self.num_heads, 0, step=-1).float().view(self.num_heads, 1, 1)
        )
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = MMLMConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)

    def _tokenize_and_encode(self, text: str) -> torch.Tensor:
        return self.tokenizer.encode(text, add_special_tokens=False, return_tensors='pt').squeeze()

    def encode_system_prompt(self, speaker_emb):
        template = self.tokenizer.apply_chat_template(
            [{"role": "system",
              "content": "Start a conversation as a helpful assistant with reference speech [REFSPEECH]"}],
            tokenize=False,
            add_generation_prompt=False,
        )
        t0, t1 = template.split("[REFSPEECH]")
        t0 = self._tokenize_and_encode(t0).to(self.lm_model.device)
        t1 = self._tokenize_and_encode(t1).to(self.lm_model.device)
        speaker_emb = speaker_emb.to(self.lm_model.device)
        return torch.cat([
            self._encode_text(t0.unsqueeze(0)),
            self.speaker_adapter(speaker_emb).unsqueeze(0),
            self._encode_text(t1.unsqueeze(0))
        ], dim=1)

    def _encode_text(self, input_ids):
        return self.lm_model.get_input_embeddings()(input_ids)

    def _process_audio_embeddings(self, input_values):
        mimi_embeds = []
        for i, head in enumerate(self.multi_decoding_head):
            embed = head(input_values[:, i, :])
            mimi_embeds.append(embed)
        # Stack embeddings: (num_heads, batch_size, seq_len, embedding_dim)
        mimi_embeds = torch.stack(mimi_embeds, dim=0)
        # Apply softmax to learned_layer_weight and weight the embeddings
        weights = F.softmax(self.learned_layer_weight, dim=0)  # (num_heads, 1, 1)
        weighted_embeds = mimi_embeds * weights[:, None, None, :]  # Broadcasting
        # Sum over heads: (batch_size, seq_len, embedding_dim)
        mimi_embeds = weighted_embeds.sum(dim=0)
        return mimi_embeds

    def _encode_audio(self, input_values):
        inputs_embeds = self.listener(input_values)
        inputs_embeds = self.listener_adapter(inputs_embeds.permute(0, 2, 1))
        return inputs_embeds

    def forward(
            self,
            input_values: Optional[torch.LongTensor] = None,  # Shape: (batch_size, num_heads, seq_len)
            listen_audio: Optional[torch.FloatTensor] = None,
            speaker_emb: Optional[torch.FloatTensor] = None,  # Shape: (batch_size, embedding_dim)
            listener_text: Optional[torch.LongTensor] = None,  # Shape: (batch_size, seq_len)
            listener_label: Optional[torch.LongTensor] = None,  # Shape: (batch_size, seq_len)
            speaker_text: Optional[torch.LongTensor] = None,  # Shape: (batch_size, seq_len)
            speaker_label: Optional[torch.LongTensor] = None,  # Shape: (batch_size, seq_len)
            codec_label: Optional[List[torch.LongTensor]] = None,  # List of labels per head
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[torch.Tensor, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Ensure that either speaker_label or listener_label is provided
        if speaker_label is None and listener_label is None:
            raise ValueError("Either 'speaker_label' or 'listener_label' must be provided.")

        if speaker_text is not None:
            speaker_text = add_bos_eos_tokens_if_not_exist(self.tokenizer, speaker_text)
        if speaker_label is not None:
            speaker_label = add_bos_eos_tokens_if_not_exist(self.tokenizer, speaker_label)

        if speaker_text is None and speaker_label is not None:
            # Shift speaker_label by 1 to create speaker_text
            speaker_text = speaker_label[:, :-1].contiguous()
            speaker_label = speaker_label[:, 1:].contiguous()
        elif speaker_label is None and speaker_text is not None:
            # Shift speaker_text by 1 to create speaker_label
            speaker_label = speaker_text[:, 1:].contiguous()
            speaker_text = speaker_text[:, :-1].contiguous()
        else:
            # Both speaker_text and speaker_label are provided
            speaker_text = speaker_text.contiguous()
            speaker_label = speaker_label.contiguous()

        # Similar processing for listener_text and listener_label
        if listener_text is not None:
            listener_text = add_bos_eos_tokens_if_not_exist(self.tokenizer, listener_text)
        if listener_label is not None:
            listener_label = add_bos_eos_tokens_if_not_exist(self.tokenizer, listener_label)

        if listener_text is None and listener_label is not None:
            listener_text = listener_label[:, :-1].contiguous()
            listener_label = listener_label[:, 1:].contiguous()
        elif listener_label is None and listener_text is not None:
            listener_label = listener_text[:, 1:].contiguous()
            listener_text = listener_text[:, :-1].contiguous()
        else:
            listener_text = listener_text.contiguous()
            listener_label = listener_label.contiguous()

        # Process embeddings
        mimi_embeds = self._process_audio_embeddings(input_values)  # (batch_size, seq_len, embedding_dim)
        audio_embeds = self._encode_audio(listen_audio)
        speaker_embeds = self._encode_text(speaker_text)  # (batch_size, seq_len, embedding_dim)
        listener_embeds = self._encode_text(listener_text)  # (batch_size, seq_len, embedding_dim)

        # Align and sum embeddings
        speak_embeds = align_and_sum_embeddings(mimi_embeds, speaker_embeds)
        listen_embeds = align_and_sum_embeddings(audio_embeds, listener_embeds)
        system_embeds = self.encode_system_prompt(speaker_emb)
        # Calculate the length of system_embeds
        system_embed_length = system_embeds.shape[1]

        # Adjust labels by prepending -100 to mask out system_embeds
        if listener_label is not None:
            listener_label = F.pad(listener_label, (system_embed_length, 0), value=-100)
        if speaker_label is not None:
            speaker_label = F.pad(speaker_label, (system_embed_length, 0), value=-100)

        combined_embeds = torch.cat([system_embeds, speak_embeds + listen_embeds], dim=1)

        # Pass through the language model
        outputs = self.lm_model(
            inputs_embeds=combined_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # Initialize loss and logits
        total_loss = None
        loss_fct = CrossEntropyLoss()

        if listener_label is not None:
            listener_logits = self.listener_head(last_hidden_state)
            listener_logits, listener_label = align_logits_and_labels(listener_logits, listener_label)
            listener_loss = loss_fct(listener_logits.view(-1, listener_logits.size(-1)), listener_label.view(-1))
            total_loss = listener_loss

        if speaker_label is not None:
            decoded_logits = self.speaker_head(last_hidden_state)
            logits, label = align_logits_and_labels(decoded_logits, speaker_label)
            lm_loss = loss_fct(logits.view(-1, logits.size(-1)), label.view(-1))
            total_loss = total_loss + lm_loss if total_loss is not None else lm_loss

        # Iterate over each decoding head
        if codec_label is not None:
            for i, decoding_head in enumerate(self.linear_decoding_head):
                if codec_label[i] is not None:
                    decoded_logits = decoding_head(last_hidden_state)
                    logits, label = align_logits_and_labels(decoded_logits, codec_label[i])
                    codec_loss = loss_fct(logits.view(-1, logits.size(-1)), label.view(-1))
                    total_loss = total_loss + codec_loss if total_loss is not None else codec_loss

        if return_dict:
            return CausalLMOutputWithPast(
                loss=total_loss,
                logits=decoded_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            return (decoded_logits, total_loss) if total_loss is not None else decoded_logits

    def generate(self, input_ids, input_values=None, listen_audio=None, speaker_emb=None, max_length=50):
        self.eval()
        generated = input_ids
        begin_gen_pos = input_ids.size(1)

        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(
                    input_values=input_values,
                    listen_audio=listen_audio,
                    speaker_emb=speaker_emb,
                    speaker_text=generated,
                )
                next_token_logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (batch_size, 1)
                generated = torch.cat([generated, next_token], dim=-1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        return generated[:, begin_gen_pos:]
