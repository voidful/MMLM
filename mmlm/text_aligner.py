import math
from pydub import AudioSegment
import torch.nn as nn
from faster_whisper import WhisperModel
from transformers import AutoTokenizer


class TextAligner(nn.Module):
    def __init__(self, model_config="deepdml/faster-whisper-large-v3-turbo-ct2", feature_extraction_interval=0.08,
                 PAD_TOKEN='[PAD]', EPAD_TOKEN='[END_PAD]'):
        super(TextAligner, self).__init__()
        self.model = WhisperModel(model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_config)
        self.feature_extraction_interval = feature_extraction_interval
        self.PAD_TOKEN = PAD_TOKEN
        self.EPAD_TOKEN = EPAD_TOKEN

        # Add PAD_TOKEN and EPAD_TOKEN to the tokenizer's vocabulary and special tokens
        self.tokenizer.add_special_tokens({
            'pad_token': self.PAD_TOKEN,
            'additional_special_tokens': [self.EPAD_TOKEN]
        })

    def forward(self, audio_path):
        # Transcribe the audio and get word-level timestamps
        segments, _ = self.model.transcribe(audio_path, word_timestamps=True)

        try:
            # Use pydub to load the audio file and calculate duration
            audio = AudioSegment.from_file(audio_path)
            audio_duration = len(audio) / 1000.0  # Duration in seconds
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return None, None

        # Calculate the sequence length based on the audio duration
        sequence_length = math.ceil((audio_duration * 1000) / 80)
        text = [self.PAD_TOKEN for _ in range(sequence_length)]

        # Create a list of frames with start times and content
        frame_list = []
        for segment in segments:
            for word in segment.words:
                frame_list.append({
                    'start_sec': word.start,
                    'content': word.word
                })

        if not frame_list:
            return None, None  # Return None if there are no frames

        start_sec = frame_list[0]['start_sec']
        raw_text = ""

        # Place words into the text list based on their timestamps
        for frame in frame_list:
            timestamp = frame['start_sec'] - start_sec
            insert_idx = int((timestamp * 1000) // 80)

            # Edge case: Discard words that are out of sequence length
            if insert_idx >= sequence_length:
                continue

            # Find the next available position if the current is occupied
            while text[insert_idx] != self.PAD_TOKEN:
                insert_idx += 1
                if insert_idx >= sequence_length:
                    break

            if insert_idx >= sequence_length:
                continue

            # Tokenize the word without adding special tokens
            text_token = self.tokenizer.tokenize(frame['content'])

            # Check if the word can fit into the remaining sequence
            if insert_idx + len(text_token) > sequence_length:
                continue

            # Place the tokens into the text list
            for i, token in enumerate(text_token):
                text[insert_idx + i] = token
            raw_text += frame['content']

        # Replace remaining PAD tokens with EPAD_TOKEN where appropriate
        for i in range(len(text) - 1):
            if text[i] == self.PAD_TOKEN and text[i + 1] != self.PAD_TOKEN:
                text[i] = self.EPAD_TOKEN

        # Handle the last token separately
        if text[-1] == self.PAD_TOKEN:
            text[-1] = self.EPAD_TOKEN

        # Convert tokens to IDs, ensuring special tokens are recognized
        token_seq = self.tokenizer.convert_tokens_to_ids(text)

        # Verify that the token sequence length matches the expected sequence length
        if len(token_seq) != sequence_length:
            print(f"Token sequence length {len(token_seq)} does not match expected sequence length {sequence_length}")
            return None, None

        # Decode the token sequence to get the text with pads
        text_with_pad = self.tokenizer.decode(token_seq, skip_special_tokens=False)
        return raw_text, text_with_pad
