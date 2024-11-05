import torch
import torch.nn as nn
from collections import deque
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


class ListenFeatureExtractor(nn.Module):
    def __init__(self, model_id="openai/whisper-large-v3-turbo",
                 sampling_rate=16000,
                 queue_duration=30,
                 step_duration=0.12):
        super(ListenFeatureExtractor, self).__init__()
        model_id = "openai/whisper-large-v3-turbo"
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=self.torch_dtype, use_safetensors=True).to(self.device)
        self.encoder = model.get_encoder()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.sampling_rate = sampling_rate
        self.queue_duration = queue_duration
        self.step_duration = step_duration
        self.num_layers = len(model.get_encoder().layers)
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers, requires_grad=True) / self.num_layers).to(
            self.device)
        self.queue_length = int(queue_duration * sampling_rate)
        self.audio_queue = deque(maxlen=self.queue_length)
        self.step_size = int(sampling_rate * step_duration)

    def forward(self, audio_array):
        feature_list = []
        for start in range(0, len(audio_array), self.step_size):
            chunk = audio_array[start:start + self.step_size]
            self.audio_queue.extend(chunk)
            feature_pos = int(
                len(self.audio_queue) / self.queue_length * 1500)  # 1500 is fixed length of 30s whisper feature
            with torch.no_grad():
                input_features = self.processor(
                    np.array(self.audio_queue), sampling_rate=self.sampling_rate, return_tensors="pt"
                ).input_features.to(self.device).to(self.torch_dtype)
                encoder_outputs = self.encoder(input_features, output_hidden_states=True).hidden_states[
                                  -self.num_layers:]  # Exclude input embedding layer
                sliced_encoder_outputs = []
                for encoder_output in encoder_outputs:
                    sliced_encoder_outputs.append(torch.mean(encoder_output[:, feature_pos:feature_pos + 4]))
                encoder_output_stack = torch.stack(sliced_encoder_outputs, dim=0)
            weighted_sum = torch.sum(self.layer_weights[:, None, None, None] * encoder_output_stack, dim=0)
            feature_list.append(weighted_sum)
        return torch.cat(feature_list, dim=1).permute(0,2,1)
