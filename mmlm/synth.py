import torch
from huggingface_hub import hf_hub_download
from moshi.models import loaders
import torch.nn as nn

from mmlm.utility import load_audio_to_tensor


class SynthFeatureExtractor(nn.Module):
    def __init__(self, num_codebooks=8):
        super(SynthFeatureExtractor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=self.device)
        mimi.set_num_codebooks(num_codebooks)
        self.mimi = mimi

    def forward(self, audio_input):
        audio_array = load_audio_to_tensor(audio_input).to(self.device)
        with torch.no_grad():
            codes = self.mimi.encode(audio_array)
            return codes.cpu()
