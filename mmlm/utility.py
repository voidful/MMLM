from typing import Optional

from pydub import AudioSegment
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


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


def align_and_sum_embeddings(voice_embeds, text_embeds):
    # Ensure both embeddings have the same sequence length
    # Assuming mimi_embeds and text_embeds are (batch_size, seq_len, embedding_dim)
    max_len = max(voice_embeds.size(1), text_embeds.size(1))
    if voice_embeds.size(1) < max_len:
        voice_embeds = F.pad(voice_embeds, (0, 0, max_len - voice_embeds.size(1), 0))
    if text_embeds.size(1) < max_len:
        text_embeds = F.pad(text_embeds, (0, 0, 0, max_len - text_embeds.size(1)))
    return voice_embeds + text_embeds


def align_logits_and_labels(logits, labels):
    logits_len = logits.size(1)
    labels_len = labels.size(1)
    if labels_len < logits_len:
        labels = F.pad(labels, (0, logits_len - labels_len), value=-100)
    elif labels_len > logits_len:
        labels = labels[:, :logits_len]
    return logits, labels


def add_bos_eos_tokens_if_not_exist(tokenizer, input_text: Optional[torch.LongTensor] = None, padding_value=0):
    """
    Add BOS and EOS tokens to each element in the input text if they do not exist.
    """
    if input_text is not None:
        if input_text.dim() == 1:
            input_text = input_text.unsqueeze(0)
        processed_texts = []
        for sequence in input_text:
            if sequence[0].item() != tokenizer.bos_token_id:
                sequence = torch.cat(
                    [torch.tensor([tokenizer.bos_token_id], dtype=torch.long, device=sequence.device), sequence])
            if sequence[-1].item() != tokenizer.eos_token_id:
                sequence = torch.cat(
                    [sequence, torch.tensor([tokenizer.eos_token_id], dtype=torch.long, device=sequence.device)])
            processed_texts.append(sequence)
        input_text = torch.nn.utils.rnn.pad_sequence(processed_texts, batch_first=True, padding_value=padding_value)
    return input_text


def load_audio_to_tensor(audio_input, sr=24000, selected_channel=None):
    """
    Load audio input into a PyTorch tensor with optional resampling and channel selection.

    Parameters:
    - audio_input: str or torch.Tensor, representing audio data or path to an audio file.
    - sr: int, the target sampling rate for resampling (default: 24000).
    - selected_channel: int, the index of the channel to select (default: None, meaning no channel selection).

    Returns:
    - A PyTorch tensor of shape (B, C, T) where B is batch size, C is number of channels, and T is time frames.
    """

    def resample_audio(audio_segment, target_sr):
        """Resample the audio to the target sampling rate."""
        return audio_segment.set_frame_rate(target_sr)

    # Load from file path
    if isinstance(audio_input, str):
        audio_segment = AudioSegment.from_file(audio_input)
        if audio_segment.frame_rate != sr:
            audio_segment = resample_audio(audio_segment, sr)
        audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        num_channels = audio_segment.channels
        audio_array = np.reshape(audio_array, (-1, num_channels)).T  # (Channels, T)
        audio_tensor = torch.tensor(audio_array).float()

    # Handle Torch Tensor directly
    elif isinstance(audio_input, torch.Tensor):
        audio_tensor = audio_input
    else:
        raise ValueError("Unsupported audio input type. Expected file path or torch.Tensor.")
    # Channel selection
    if selected_channel is not None:
        if selected_channel >= audio_tensor.shape[0]:
            raise ValueError(
                f"Selected channel {selected_channel} is out of range for audio with {audio_tensor.shape[0]} channels."
            )
        audio_tensor = audio_tensor[selected_channel:selected_channel + 1]  # Select specific channel
    else:
        # take the average of all channels
        audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
    # Dimension adjustment to (B, C, T)
    if audio_tensor.dim() == 1:  # (T)
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # (B=1, C=1, T)
    elif audio_tensor.dim() == 2:  # (C, T)
        audio_tensor = audio_tensor.unsqueeze(0)  # (B=1, C, T)
    elif audio_tensor.dim() > 3:
        raise ValueError("Audio input has unsupported dimensions after processing, expected 1D or 2D tensor, got "
                         f"{audio_tensor.dim()}D tensor.", audio_tensor.shape)

    return audio_tensor
