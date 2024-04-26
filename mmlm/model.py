from typing import Optional, Union, Tuple

from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from embedder import export_embedder
import torch
import torch.nn.functional as F


class MMLM(nn.Module):
    def __init__(
        self,
        lm_config,
        lm_model=None,
        lm_tokenizer=None,
        audio_config=1,
        audio_model=None,
        audio_adapter_config=None,
        visual_config=1,
        visual_model=None,
        visual_adapter_config=None,
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Language Model Setup
        self.lm_model = (
            lm_model.to(self.device)
            if lm_model is not None
            else AutoModelForCausalLM.from_pretrained(lm_config).to(self.device)
        )
        self.tokenizer = (
            lm_tokenizer
            if lm_tokenizer
            else AutoTokenizer.from_pretrained(lm_config)
        )

        # Modality Configurations
        self.audio_config = audio_config
        self.visual_config = visual_config

        # Audio Feature Processing
        if isinstance(audio_config, int):
            self._setup_discrete_feature_weights(audio_config, "audio")
        else:
            self._setup_continuous_feature_processing(
                audio_config, audio_model, audio_adapter_config, "audio"
            )

        # Visual Feature Processing
        if isinstance(visual_config, int):
            self._setup_discrete_feature_weights(visual_config, "visual")
        else:
            self._setup_continuous_feature_processing(
                visual_config, visual_model, visual_adapter_config, "visual"
            )

    def _setup_discrete_feature_weights(self, config, modality):
        learnable_weight_init = torch.arange(config, 0, step=-1).float().view(config, 1)
        setattr(self, f"{modality}_learnable_weight", nn.Parameter(learnable_weight_init))

    def _setup_continuous_feature_processing(self, config, model, adapter_config, modality):
        model = model.to(self.device) if model is not None else AutoModel.from_pretrained(config)
        setattr(self, f"{modality}_model", model)
        setattr(
            self,
            f"{modality}_adapter",
            export_embedder[adapter_config](
                input_size=model.config.hidden_size, output_size=self.lm_model.config.hidden_size
            ).to(self.device),
        )

    def continue_audio_feature_type_ids(self):
        base_tag_id = self.tokenizer.encode(f"CAUDIO_TAG_{0}")[0]
        return [base_tag_id + u for u in range(100)]

    def continue_visual_feature_type_ids(self):
        base_tag_id = self.tokenizer.encode(f"CVISUAL_TAG_{0}")[0]
        return [base_tag_id + u for u in range(100)]

    def discrete_audio_feature_type_ids(self):
        base_tag_id = self.tokenizer.encode(f"a_tok_{0}")[0]
        return [base_tag_id + u for u in range(1024 * 10)]

    def discrete_visual_feature_type_ids(self):
        base_tag_id = self.tokenizer.encode(f"v_tok_{0}")[0]
        return [base_tag_id + u for u in range(1024 * 10)]

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            audio_features=None,
            vision_features=None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
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
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

        if inputs_embeds is None:
            embeder = self.lm_model.get_input_embeddings()
            inputs_embeds = []
            for batch_num, batch_input in enumerate(input_ids):
                audio_discrete_token = []
                visual_discrete_token = []
                text_ids = []
                input_embeds = []
                for i in batch_input:
                    if i in self.discrete_audio_feature_type_ids():
                        audio_discrete_token.append(i)
                        if text_ids:
                            input_embeds.append(embeder(torch.LongTensor(text_ids).to(self.device)))
                        text_ids = []
                    elif i in self.discrete_visual_feature_type_ids():
                        visual_discrete_token.append(i)
                        if text_ids:
                            input_embeds.append(embeder(torch.LongTensor(text_ids).to(self.device)))
                        text_ids = []
                    else:
                        text_ids.append(i)
                        if len(audio_discrete_token) > 0:
                            discrete_audio_input_id = torch.tensor(audio_discrete_token).view(self.audio_config, -1)
                            discrete_audio_input_ids = []
                            for i in range(self.audio_config):
                                input_scale = embeder(discrete_audio_input_id[i, :].to(self.device))
                                discrete_audio_input_ids.append(input_scale)
                            weighted_discrete_inputs_embeds = torch.mul(
                                torch.stack(discrete_audio_input_ids, dim=0).to(self.device),
                                F.softmax(self.audio_learnable_weight, dim=0).to(self.device))
                            weighted_discrete_inputs_embeds = torch.sum(weighted_discrete_inputs_embeds, dim=0)
                            if discrete_audio_input_ids:
                                input_embeds.append(weighted_discrete_inputs_embeds)
                            audio_discrete_token = []
                        elif len(visual_discrete_token) > 0:
                            discrete_visual_input_id = torch.tensor(visual_discrete_token).view(self.visual_config, -1)
                            discrete_visual_input_ids = []
                            for i in range(self.visual_config):
                                input_scale = embeder(discrete_visual_input_id[i, :].to(self.device))
                                discrete_visual_input_ids.append(input_scale)
                            weighted_discrete_inputs_embeds = torch.mul(
                                torch.stack(discrete_visual_input_ids, dim=0).to(self.device),
                                F.softmax(self.visual_learnable_weight, dim=0).to(self.device))
                            weighted_discrete_inputs_embeds = torch.sum(weighted_discrete_inputs_embeds, dim=0)
                            if discrete_visual_input_ids:
                                input_embeds.append(weighted_discrete_inputs_embeds)
                            visual_discrete_token = []
                inputs_embeds.append(torch.cat(input_embeds))
            inputs_embeds = torch.stack(inputs_embeds)
            outputs = self.lm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif input_ids.shape[-1] == 1:
            outputs = self.lm_model(
                input_ids=input_ids,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif self.audio_config or self.visual_config:  # repack input_embeds
            for batch_num, batch_input in enumerate(input_ids):
                vision_features_id = 0
                audio_features_id = 0
                for pos, ids in enumerate(batch_input):
                    if ids in self.continue_audio_feature_type_ids():
                        audio_feature = self.audio_adapter(audio_features[batch_num][audio_features_id]).to(self.device)
                        audio_features_id += 1
                        inputs_embeds = torch.cat(
                            (inputs_embeds[:, :pos, :], audio_feature, inputs_embeds[:, pos + 1:, :]), dim=1).to(
                            self.device)
                    if ids in self.continue_visual_feature_type_ids():
                        vision_features = self.visual_adapter(vision_features[batch_num][vision_features_id])
                        vision_features_id += 1
                        inputs_embeds = torch.cat(
                            (inputs_embeds[:, :pos, :], vision_features, inputs_embeds[:, pos + 1:, :]), dim=1)
            outputs = self.lm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
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
        device = next(self.parameters()).device
        generated = input_ids.to(self.device)
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(input_ids=generated, audio_features=audio_feature)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_logits = next_token_logits
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat((generated, next_token), dim=-1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        return generated
