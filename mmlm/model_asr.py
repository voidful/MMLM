from typing import Optional, Union, Tuple
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, AutoTokenizer
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.configuration_utils import PretrainedConfig
import torch
import torch.nn.functional as F
from mmlm.listener import ListenFeatureExtractor


class MMLMASRConfig(PretrainedConfig):
    model_type = "mmlmasr"

    def __init__(self, lm_model_name="voidful/SmolLM2-360M-Instruct-ASR", **kwargs):
        super().__init__(**kwargs)
        self.lm_model_name = lm_model_name


class MMLMASR(PreTrainedModel):
    config_class = MMLMASRConfig

    def __init__(self, config: MMLMASRConfig):
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                f"Parameter config in `MMLMASR(config)` should be an instance of class `PretrainedConfig`, "
                f"but got {type(config)} instead. To create a model from a pretrained model, use "
                f"`MMLMASR.from_pretrained(PRETRAINED_MODEL_NAME)`."
            )
        super().__init__(config)
        # Language Model Setup
        self.lm_model = AutoLigerKernelForCausalLM.from_pretrained(
            config.lm_model_name,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2", device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.listener = ListenFeatureExtractor()
        self.adapter = nn.Linear(512, self.lm_model.get_input_embeddings().weight.shape[-1])

        # Freeze LM parameters
        for param in self.lm_model.parameters():
            param.requires_grad = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = MMLMASRConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Enable gradient checkpointing for the model. This modifies the behavior of the internal layers
        to reduce memory usage at the cost of additional computation during the backward pass.
        """
        self.lm_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        if hasattr(self.listener, "gradient_checkpointing_enable"):
            self.listener.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """
        Disable gradient checkpointing for the model.
        """
        self.lm_model.gradient_checkpointing_disable()
        if hasattr(self.listener, "gradient_checkpointing_disable"):
            self.listener.gradient_checkpointing_disable()

    def encode_text(self, input_ids):
        embeder = self.lm_model.get_input_embeddings()
        # Create input_ids and labels for causal language model
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        embedding = embeder(input_ids)
        return embedding, input_ids, labels

    def encode_audio(self, input_values):
        inputs_embeds = self.listener(input_values)
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode audio
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

        # Sum audio and text embeddings
        inputs_embeds = audio_embeds + text_embeds

        # Forward pass through LM
        outputs = self.lm_model(
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits
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
        generated = input_ids
        begin_gen_pos = input_ids.shape[1]
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(input_values=audio_feature, labels=generated)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat((generated, next_token), dim=-1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        return generated[:, begin_gen_pos:-1]
