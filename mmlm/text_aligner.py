import math
import librosa
import torch.nn as nn
from faster_whisper import WhisperModel


class TextAligner(nn.Module):
    def __init__(self, model_config="deepdml/faster-whisper-large-v3-turbo-ct2", feature_extraction_interval=0.12):
        super(TextAligner, self).__init__()
        self.model = WhisperModel(model_config)
        self.feature_extraction_interval = feature_extraction_interval

    def insert_end_pads(self, aligned_features):
        """Insert 'end_pad' markers between words and pad sequences where needed, except after the last word."""
        i = 0
        while i < len(aligned_features):
            if aligned_features[i] == '[PAD]':
                j = i
                while j + 1 < len(aligned_features) and aligned_features[j + 1] == '[PAD]':
                    j += 1

                # Assign 'end_pad' if a word follows the current pad sequence
                if j + 1 < len(aligned_features) and aligned_features[j + 1] not in ['[PAD]', '[END_PAD]']:
                    aligned_features[j] = '[END_PAD]'
                elif j + 1 < len(aligned_features) and aligned_features[j + 1] == '[PAD]':
                    aligned_features[j] = '[PAD]'
                i = j + 1
            elif i + 1 < len(aligned_features) and aligned_features[i + 1] not in ['[PAD]', '[END_PAD]']:
                # Add end pad between words, except for the last word
                aligned_features[i] += ' [END_PAD]'
                i += 1
            else:
                i += 1

    def map_word_to_intervals(self, word, num_features):
        """Map a word to the correct feature interval index range."""
        start_index = int(word.start // self.feature_extraction_interval)
        end_index = int(word.end // self.feature_extraction_interval)

        # Ensure indices are within bounds
        start_index = max(0, min(start_index, num_features - 1))
        end_index = max(0, min(end_index, num_features - 1))

        return start_index, end_index

    def forward(self, audio_path):
        segments, _ = self.model.transcribe(audio_path, word_timestamps=True)
        audio_duration = librosa.get_duration(path=audio_path)
        num_features = math.ceil(audio_duration / self.feature_extraction_interval)
        aligned_features = ['[PAD]'] * num_features

        for segment in segments:
            for word in segment.words:
                start_index, end_index = self.map_word_to_intervals(word, num_features)
                aligned_features[start_index] = word.word

        self.insert_end_pads(aligned_features)
        return aligned_features
