import os
import math
import pandas as pd
import json
from pydub import AudioSegment
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
import logging
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("process.log")]
)

# Global variable for tokenizer
global_tokenizer = None

def init_tokenizer(model_config):
    """
    Initialize the global tokenizer.
    This function is called once at the start of each process.
    """
    global global_tokenizer
    if global_tokenizer is None:
        logging.info("Initializing tokenizer...")
        global_tokenizer = AutoTokenizer.from_pretrained(model_config)
        global_tokenizer.add_special_tokens({
            'pad_token': '[PAD]',
            'additional_special_tokens': ['[END_PAD]']
        })


class TextAligner:
    def __init__(self, feature_extraction_interval=0.08,
                 PAD_TOKEN='[PAD]', EPAD_TOKEN='[END_PAD]'):
        global global_tokenizer
        self.tokenizer = global_tokenizer  # Use the shared tokenizer
        self.feature_extraction_interval = feature_extraction_interval
        self.PAD_TOKEN = PAD_TOKEN
        self.EPAD_TOKEN = EPAD_TOKEN

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

        if word_df.empty:
            logging.warning(f"The word CSV {word_csv_path} is empty.")
            return None, None

        try:
            word_df[start_col] = word_df[start_col].astype(str).str.replace('s', '').astype(float)
            start_sec = word_df[start_col].iloc[0]
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
    try:
        word_csv_path, segment_csv_path, wav_path, model_config = params
        init_tokenizer(model_config)  # Ensure the tokenizer is initialized in the process
        aligner = TextAligner()
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
    except Exception as e:
        logging.error(f"Error in process_single_file: {e}")
        return None


def process_file(args):
    word_dir, segment_dir, wav_dir, output_path = args.word_dir, args.segment_dir, args.wav_dir, args.output_path

    def find_files(directory, extension):
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(extension):
                    yield os.path.join(root, file)

    word_files = list(find_files(word_dir, '.csv'))
    segment_files = list(find_files(segment_dir, '.csv'))
    audio_files = list(find_files(wav_dir, ('.m4a', '.wav', '.mp3')))

    word_map = {os.path.splitext(os.path.basename(f))[0]: f for f in word_files}
    segment_map = {os.path.splitext(os.path.basename(f))[0]: f for f in segment_files}
    audio_map = {os.path.splitext(os.path.basename(f))[0]: f for f in audio_files}

    file_list = []
    for name in word_map:
        if name in segment_map and name in audio_map:
            file_list.append((word_map[name], segment_map[name], audio_map[name], args.model_config))

    total_partitions = args.partition_total
    partition_index = args.partition_index - 1
    file_list = file_list[partition_index::total_partitions]

    processed_files = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as output_file:
            for line in output_file:
                try:
                    processed_files.add(json.loads(line)['audio_path'])
                except json.JSONDecodeError:
                    continue

    file_list = [f for f in file_list if f[2] not in processed_files]

    logging.info(f"Processing partition {args.partition_index}/{args.partition_total}")
    logging.info(f"Total files to process in this partition: {len(file_list)}")

    with open(output_path, 'a', encoding='utf-8') as output_file:
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            for result in tqdm(executor.map(process_single_file, file_list), total=len(file_list)):
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
    parser.add_argument("--partition", type=str, default="1/1", help="Partition in the format index/total (e.g., 1/10).")

    args = parser.parse_args()

    partition_parts = args.partition.split('/')
    args.partition_index = int(partition_parts[0])
    args.partition_total = int(partition_parts[1])

    init_tokenizer(args.model_config)  # Initialize tokenizer once in the main process
    process_file(args)


if __name__ == "__main__":
    main()
