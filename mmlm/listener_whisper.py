import torch.nn as nn
from collections import deque
import numpy as np
import torch
from transformers import AutoModel, AutoFeatureExtractor

from mmlm.utility import load_audio_to_tensor


class ListenFeatureExtractor(nn.Module):
    def __init__(self, model_id="openai/whisper-large-v3-turbo",
                 sampling_rate=16000,
                 encode_feature_size=1500,  # Fixed length of 30s whisper feature
                 queue_duration=30,
                 step_duration=0.08):
        super(ListenFeatureExtractor, self).__init__()
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModel.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=self.torch_dtype, use_safetensors=True).to(self.device)
        num_layers = 0
        for module in model.modules():
            if isinstance(module, nn.ModuleList):
                num_layers = len(module)
                break
        try:
            self.encoder = model.get_encoder()
        except:
            self.encoder = model
        self.processor = AutoFeatureExtractor.from_pretrained(model_id)
        self.sampling_rate = sampling_rate
        self.queue_duration = queue_duration
        self.step_duration = step_duration

        self.num_layers = num_layers
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers, requires_grad=True) / self.num_layers).to(
            self.device)
        self.queue_length = int(queue_duration * sampling_rate)
        self.audio_queue = deque(maxlen=self.queue_length)
        self.step_size = int(sampling_rate * step_duration)
        self.encode_feature_size = encode_feature_size

    def forward(self, audio_input):
        audio_arrays = load_audio_to_tensor(audio_input)
        audio_arrays = torch.mean(audio_arrays, dim=1)  # Collapse channels if stereo

        batch_features = []
        max_len = 0  # Track max length for padding

        for audio_array in audio_arrays:
            feature_list = []
            for start in range(0, audio_array.shape[-1], self.step_size):
                chunk = audio_array[start:start + self.step_size]

                # Skip processing for chunks that are all zero (padded)
                if torch.all(chunk == 0):
                    continue

                self.audio_queue.extend(chunk.tolist())

                feature_pos = int(
                    len(self.audio_queue) / self.queue_length * self.encode_feature_size)
                with torch.no_grad():
                    input_features = self.processor(
                        np.array(self.audio_queue), sampling_rate=self.sampling_rate, return_tensors="pt"
                    ).input_features.to(self.device).to(self.torch_dtype)
                    encoder_outputs = self.encoder(input_features, output_hidden_states=True).hidden_states[
                                      -self.num_layers:]  # Exclude input embedding layer
                    sliced_encoder_outputs = []
                    for encoder_output in encoder_outputs:
                        sliced_mean = torch.mean(
                            encoder_output[:, feature_pos:feature_pos + 4, :])  # Mean over sequence length
                        sliced_encoder_outputs.append(sliced_mean)
                    encoder_output_stack = torch.stack(sliced_encoder_outputs, dim=0)

                # Compute weighted sum for the chunk
                weighted_sum = torch.sum(self.layer_weights[:, None, None] * encoder_output_stack, dim=0)
                feature_list.append(weighted_sum)

            # Stack features for this audio input and track max length
            if feature_list:
                audio_features = torch.stack(feature_list, dim=-1)
                max_len = max(max_len, audio_features.shape[-1])
                batch_features.append(audio_features)

        # Pad each feature in batch_features to the max_len
        for i in range(len(batch_features)):
            if batch_features[i].shape[-1] < max_len:
                padding_size = max_len - batch_features[i].shape[-1]
                batch_features[i] = torch.cat([batch_features[i],
                                               torch.zeros(batch_features[i].shape[0], batch_features[i].shape[1],
                                                           padding_size, device=self.device)], dim=-1)

        # Stack all batch features for each input
        return torch.stack(batch_features, dim=0).squeeze(1)
