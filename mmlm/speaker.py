import torch
from speechbrain.inference.speaker import EncoderClassifier
import logging

logging.getLogger('speechbrain').setLevel(logging.WARNING)


class ECAPATDNN:
    def __init__(self, device='cuda'):
        self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                         run_opts={"device": device})
        self.sr = 16000  # Sampling rate (16kHz)

    def encode(self, audios):
        target_length = 5 * self.sr  # Target length: 5 seconds (80000 samples at 16kHz)

        # Truncate audios longer than 5 seconds
        processed_audios = [audio[:target_length] for audio in audios]

        # Pad to the longest length in the batch (if required)
        padded = torch.nn.utils.rnn.pad_sequence(processed_audios, batch_first=True, padding_value=0)

        # Compute `wav_lens` based on original lengths before padding
        wav_lens = torch.tensor([min(len(audio), target_length) / target_length for audio in audios])

        # Compute embeddings
        embeddings = self.classifier.encode_batch(padded, wav_lens=wav_lens).cpu()
        return embeddings
