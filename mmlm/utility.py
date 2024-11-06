import librosa
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def load_audio_to_tensor(audio_input):
    # Check if audio_input is a file path, numpy array, or list and convert accordingly
    if isinstance(audio_input, str):  # If input is a file path
        audio_array, _ = librosa.load(audio_input, sr=None)  # Preserve original sampling rate
        audio_array = torch.tensor(audio_array).float()
    elif isinstance(audio_input, np.ndarray):
        audio_array = torch.from_numpy(audio_input).float()
    elif isinstance(audio_input, list):
        # Convert each element to tensor and check lengths for padding
        audio_tensors = [torch.tensor(arr).float() for arr in audio_input]
        if all(t.dim() == 1 for t in audio_tensors):  # If all are 1D tensors (single-channel audio)
            audio_array = pad_sequence(audio_tensors, batch_first=True)  # Pad to [batch, max_length]
        else:
            raise ValueError("All elements in the list must be 1D tensors or numpy arrays.")
    else:
        raise ValueError("Unsupported audio input type")

    # Check the dimensions and adjust if necessary
    if audio_array.dim() == 1:  # Shape is [wav_length]
        audio_array = audio_array.unsqueeze(0).unsqueeze(0)  # Expand to [1, 1, wav_length]
    elif audio_array.dim() == 2:  # Shape is [batch, max_length]
        audio_array = audio_array.unsqueeze(1)  # Expand to [batch, 1, max_length] for single-channel batch
    elif audio_array.dim() == 3:  # Shape is [batch, channel, wav_length]
        # No further adjustment needed for a 3D batch
        pass
    else:
        raise ValueError("Audio input has unsupported dimensions after processing")

    return audio_array
