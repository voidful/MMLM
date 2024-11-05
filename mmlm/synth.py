import moshi
import librosa
import torch
import soundfile as sf
from huggingface_hub import hf_hub_download
from moshi.models import loaders
import torch.nn as nn
import numpy as np


class SynthFeatureExtractor(nn.Module):
    def __init__(self, num_codebooks=8):
        super(SynthFeatureExtractor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=self.device)
        mimi.set_num_codebooks(num_codebooks)
        self.mimi = mimi

    def forward(self, audio_array):
        # Convert input to a tensor if it's a numpy array or list
        if isinstance(audio_array, np.ndarray):
            audio_array = torch.from_numpy(audio_array).to(self.device)
        elif isinstance(audio_array, list):
            audio_array = torch.tensor(audio_array).to(self.device)

        # Check the dimensions and adjust if necessary
        if audio_array.dim() == 1:  # Shape is [wav_length]
            audio_array = audio_array.unsqueeze(0).unsqueeze(0)  # Expand to [1, 1, wav_length]

        elif audio_array.dim() == 2:  # Shape is [channel, wav_length]
            audio_array = audio_array.unsqueeze(0)  # Expand to [1, channel, wav_length]

        with torch.no_grad():
            codes = self.mimi.encode(audio_array)
            return codes.cpu()
