import os
import math
import pandas as pd
import json
from pydub import AudioSegment
from transformers import AutoTokenizer
import argparse
import logging
from multiprocessing import cpu_count
from tqdm.contrib.concurrent import process_map

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[logging.StreamHandler()]
)

class TextAligner:
    def __init__(self, model_config="deepdml/faster-whisper-large-v3-turbo-ct2",
                 feature_extraction_interval=0.08,
                 PAD_TOKEN='[PAD]', EPAD_TOKEN='[END_PAD]'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_config)
        self.feature_extraction_interval = feature_extraction_interval
        self.PAD_TOKEN = PAD_TOKEN
        self.EPAD_TOKEN = EPAD_TOKEN

        # Add PAD_TOKEN and EPAD_TOKEN to the tokenizer's vocabulary and special tokens
        self.tokenizer.add_special_tokens({
            'pad_token': self.PAD_TOKEN,
            'additional_special_tokens': [self.EPAD_TOKEN]
        })

    def align_words(self, audio_path, word_csv_path):
        try:
            audio = AudioSegment.from_file(audio_path)
            audio_duration = len(audio) / 1000.0  # Duration in seconds
        except Exception as e:
            logging.warning(f"Failed to load audio file {audio_path}: {e}")
            return None, None

        sequence_length = math.ceil((audio_duration * 1000) / 80)
        text = [self.PAD_TOKEN for _ in range(sequence_length)]

        try:
            word_df = pd.read_csv(word_csv_path, keep_default_na=False)
        except Exception as e:
            logging.warning(f"Failed to read word CSV {word_csv_path}: {e}")
            return None, None

        # Validate columns
        start_col = next((col for col in ['start', 'start_time'] if col in word_df.columns), None)
        word_col = next((col for col in ['word', 'text'] if col in word_df.columns), None)

        if not start_col or not word_col:
            logging.warning(f"Word CSV {word_csv_path} does not contain required columns ('start' or 'word').")
            return None, None

        # Check if the DataFrame is empty
        if word_df.empty:
            logging.warning(f"The word CSV {word_csv_path} is empty.")
            return None, None

        try:
            # Clean up and process the 'start' column
            word_df[start_col] = word_df[start_col].astype(str).str.replace('s', '').astype(float)
            start_sec = word_df[start_col].iloc[0]  # This should now be safe
        except Exception as e:
            logging.warning(f"Failed to process 'start' column in {word_csv_path}: {e}")
            return None, None

        raw_text = ""
        for _, row in word_df.iterrows():
            timestamp = row[start_col] - start_sec
            insert_idx = int((timestamp * 1000) // 80)
            if insert_idx >= sequence_length:
                continue

            while insert_idx < sequence_length and text[insert_idx] != self.PAD_TOKEN:
                insert_idx += 1

            if insert_idx >= sequence_length:
                continue

            word_text = str(row[word_col])
            text_tokens = self.tokenizer.tokenize(word_text)
            if insert_idx + len(text_tokens) > sequence_length:
                continue

            for i, token in enumerate(text_tokens):
                text[insert_idx + i] = token
            raw_text += word_text

        for i in range(len(text) - 1):
            if text[i] == self.PAD_TOKEN and text[i + 1] != self.PAD_TOKEN:
                text[i] = self.EPAD_TOKEN
        if text[-1] == self.PAD_TOKEN:
            text[-1] = self.EPAD_TOKEN

        token_seq = self.tokenizer.convert_tokens_to_ids(text)
        if len(token_seq) != sequence_length:
            logging.warning(f"Token sequence length mismatch for {audio_path}")
            return None, None

        text_with_pad = self.tokenizer.decode(token_seq, skip_special_tokens=False)
        return raw_text, text_with_pad


def process_single_file(params):
    word_csv_path, segment_csv_path, wav_path, model_config = params
    aligner = TextAligner(model_config=model_config)
    raw_text, text_with_pad = aligner.align_words(wav_path, word_csv_path)

    if raw_text is not None and text_with_pad is not None:
        try:
            segment_df = pd.read_csv(segment_csv_path)
            text_field = ' '.join(segment_df['text'].astype(str).tolist()) if 'text' in segment_df.columns else ""
        except Exception as e:
            logging.warning(f"Failed to read segment CSV {segment_csv_path}: {e}")
            text_field = ""

        return {
            "audio_path": wav_path,
            "text_with_pad": text_with_pad,
            "text": text_field
        }
    return None


def process_file(args):
    word_dir, segment_dir, wav_dir, output_path = args.word_dir, args.segment_dir, args.wav_dir, args.output_path

    # Function to recursively find all files in a directory
    def find_files(directory, extension):
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(extension):
                    yield os.path.join(root, file)

    # Find all relevant files in each directory
    word_files = list(find_files(word_dir, '.csv'))
    segment_files = list(find_files(segment_dir, '.csv'))
    audio_files = list(find_files(wav_dir, ('.m4a', '.wav', '.mp3')))

    # Match files by name (without directory or extension)
    word_map = {os.path.splitext(os.path.basename(f))[0]: f for f in word_files}
    segment_map = {os.path.splitext(os.path.basename(f))[0]: f for f in segment_files}
    audio_map = {os.path.splitext(os.path.basename(f))[0]: f for f in audio_files}

    # Collect file pairs
    file_list = []
    for name in word_map:
        if name in segment_map and name in audio_map:
            file_list.append((word_map[name], segment_map[name], audio_map[name], args.model_config))

    logging.info(f"Total files to process: {len(file_list)}")

    with open(output_path, 'w', encoding='utf-8') as output_file:
        for result in process_map(process_single_file, file_list, max_workers=cpu_count(), chunksize=1):
            if result:
                json.dump(result, output_file, ensure_ascii=False)
                output_file.write('\n')
                output_file.flush()


def main():
    parser = argparse.ArgumentParser(description="Align text to audio features.")
    parser.add_argument("--word_dir", type=str, required=True, help="Directory containing word CSV files.")
    parser.add_argument("--segment_dir", type=str, required=True, help="Directory containing segment CSV files.")
    parser.add_argument("--wav_dir", type=str, required=True, help="Directory containing audio files.")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSONL file path.")
    parser.add_argument("--model_config", type=str, default="deepdml/faster-whisper-large-v3-turbo-ct2", help="Model configuration.")

    args = parser.parse_args()
    process_file(args)


if __name__ == "__main__":
    main()
