import librosa
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

import torch
from torch.nn.utils.rnn import pad_sequence


class MMLMUtility():
    def __init__(self, mmlm_model):
        self.mmlm_model = mmlm_model

    def tokenize_function(self, examples):
        model_inputs = self.mmlm_model.tokenizer(examples['input'] + examples['label'])
        labels = self.mmlm_model.tokenizer(examples['label'] + self.mmlm_model.tokenizer.eos_token)
        padding_size = len(model_inputs['input_ids']) - len(labels["input_ids"])
        model_inputs["label_ids"] = [-100] * padding_size + labels["input_ids"]
        return model_inputs

    class MMLMDataCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, features):
            return {
                'input_ids': pad_sequence([torch.tensor(i['input_ids']) for i in features], batch_first=True,
                                          padding_value=self.tokenizer.eos_token_id),
                'labels': pad_sequence([torch.tensor(i['label_ids']) for i in features], batch_first=True,
                                       padding_value=-100),
            }


import librosa
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def load_audio_to_tensor(audio_input, sr=24000, selected_channel=None):
    """
    Load audio input into a PyTorch tensor with optional resampling and channel selection.

    Parameters:
    - audio_input: str, np.ndarray, list, or torch.Tensor representing audio data or path to an audio file.
    - sr: int, the target sampling rate for resampling (default: 24000).
    - selected_channel: int, the index of the channel to select (default: None, meaning no channel selection).

    Returns:
    - A PyTorch tensor of shape (B, C, T) where B is batch size, C is number of channels, and T is time frames.
    """

    def resample_if_needed(audio, orig_sr, target_sr):
        """Resample the audio only if the original and target sampling rates differ."""
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr) if orig_sr != target_sr else audio

    # Load from file path
    if isinstance(audio_input, str):
        audio_array, orig_sr = librosa.load(audio_input, sr=None, mono=False)
        if sr is not None:
            audio_array = resample_if_needed(audio_array, orig_sr, sr)
        audio_array = torch.tensor(audio_array).float()

    # Handle NumPy array input
    elif isinstance(audio_input, np.ndarray):
        if sr is not None:
            audio_input = resample_if_needed(audio_input, orig_sr=librosa.get_samplerate(audio_input), target_sr=sr)
        audio_array = torch.from_numpy(audio_input).float()

    # Handle list of arrays or lists
    elif isinstance(audio_input, list):
        audio_tensors = [
            torch.tensor(
                resample_if_needed(arr, librosa.get_samplerate(arr), sr)).float() if sr is not None else torch.tensor(
                arr).float()
            for arr in audio_input
        ]
        if all(t.dim() == 1 for t in audio_tensors):
            audio_array = pad_sequence(audio_tensors, batch_first=True)
        else:
            raise ValueError("All elements in the list must be 1D tensors or numpy arrays.")

    # Handle Torch Tensor directly
    elif isinstance(audio_input, torch.Tensor):
        audio_array = audio_input
        if sr is not None and audio_array.dim() == 1:
            audio_array = torch.tensor(
                resample_if_needed(audio_array.numpy(), orig_sr=librosa.get_samplerate(audio_array), target_sr=sr)
            ).float()
    else:
        raise ValueError("Unsupported audio input type")

    # Channel selection
    if audio_array.dim() == 2:  # (Channels, T)
        if selected_channel is not None:
            if selected_channel >= audio_array.shape[0]:
                raise ValueError(
                    f"Selected channel {selected_channel} is out of range for audio with {audio_array.shape[0]} channels.")
            audio_array = audio_array[selected_channel:selected_channel + 1]  # Select specific channel
    elif audio_array.dim() == 3:  # (Batch, Channels, T)
        if selected_channel is not None:
            if selected_channel >= audio_array.shape[1]:
                raise ValueError(
                    f"Selected channel {selected_channel} is out of range for audio with {audio_array.shape[1]} channels.")
            audio_array = audio_array[:, selected_channel:selected_channel + 1, :]  # Select specific channel

    # Dimension adjustment to (B, C, T)
    if audio_array.dim() == 1:  # (T)
        audio_array = audio_array.unsqueeze(0).unsqueeze(0)  # (B=1, C=1, T)
    elif audio_array.dim() == 2:  # (C, T) or (T, C)
        audio_array = audio_array.unsqueeze(0)  # (B=1, C, T)
    elif audio_array.dim() > 3:
        raise ValueError("Audio input has unsupported dimensions after processing.")

    return audio_array
