import librosa
import numpy as np
import torch


def load_audio_to_tensor(audio_input):
    # Check if audio_input is a file path, numpy array, or list and convert accordingly
    if isinstance(audio_input, str):  # If input is a file path
        audio_array, _ = librosa.load(audio_input, sr=None)  # Preserve original sampling rate
        audio_array = torch.tensor(audio_array).float()
    elif isinstance(audio_input, np.ndarray):
        audio_array = torch.from_numpy(audio_input)
    elif isinstance(audio_input, list):
        audio_array = torch.tensor(audio_input)
    else:
        raise ValueError("Unsupported audio input type")

    # Check the dimensions and adjust if necessary
    if audio_array.dim() == 1:  # Shape is [wav_length]
        audio_array = audio_array.unsqueeze(0).unsqueeze(0)  # Expand to [1, 1, wav_length]
    elif audio_array.dim() == 2:  # Shape is [channel, wav_length]
        audio_array = audio_array.unsqueeze(0)  # Expand to [1, channel, wav_length]

    return audio_array
