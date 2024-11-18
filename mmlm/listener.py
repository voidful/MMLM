import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mimi.modeling_mimi import MimiModel

from mmlm.utility import load_audio_to_tensor


class ListenFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        mimi_model = MimiModel.from_pretrained("kyutai/mimi").to(device)
        for param in mimi_model.parameters():
            param.requires_grad = False
        self.embeddings = mimi_model.encoder.to(device)
        self.model = mimi_model.encoder_transformer.to(device)
        self.downsample = mimi_model.downsample.to(device)
        for param in self.embeddings.parameters():
            param.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.downsample.parameters():
            param.requires_grad = False

        self.layer_outputs = []
        self.layer_weights = nn.Parameter(torch.ones(len(self.model.layers)))

        # Register a hook on each layer to capture its downsampled output
        for layer in self.model.layers:
            layer.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        """Hook to capture the downsampled output of each layer."""
        downsampled_output = self.downsample(output[0].transpose(1, 2)).transpose(1, 2)  # Downsample and adjust shape
        self.layer_outputs.append(downsampled_output)

    def forward(self, audio_input):
        audio_input = audio_input.to(self.device)
        # Clear previous layer outputs
        self.layer_outputs = []

        # Prepare audio input tensor
        audio_array = load_audio_to_tensor(audio_input, 24000).to(self.device)

        # Get embeddings and pass them through the transformer layers
        embeddings = self.embeddings(audio_array)
        _ = self.model(embeddings.transpose(1, 2), past_key_values=None)

        # Stack downsampled outputs and apply weights
        layer_outputs = torch.stack(self.layer_outputs,
                                    dim=0)  # Shape: (num_layers, batch, sequence_length, hidden_dim)
        weights = F.softmax(self.layer_weights, dim=0)  # Normalize weights across layers
        # Perform weighted sum across the layer dimension (dim=0)
        weighted_sum = torch.sum(weights[:, None, None, None] * layer_outputs,
                                 dim=0)  # Shape: (batch, sequence_length, hidden_dim)
        return weighted_sum.transpose(1, 2)
