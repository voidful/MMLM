import os
import logging
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
import wandb
from datasets import load_dataset
from mmlm.model_tts import MMLMTTSConfig, MMLMTTS
import numpy as np

# ========================
# Global Configuration
# ========================
WANDB_PROJECT_NAME = "mmlm-conv"
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
DATASET = load_dataset("voidful/conv_test_data")['train']
LM_MODEL_NAME = "voidful/SmolLM2-360M-Instruct-Whisper"
OUTPUT_DIR = "./mmlm-conv-training"
MODEL_SAVE_PATH = "./mmlm-conv-model"
TRAIN_TEST_SPLIT_RATIO = 0.1
EPOCHS = 5
BATCH_SIZE = 1
LEARNING_RATE = 5e-5
GRADIENT_ACCUMULATION_STEPS = 50
USE_BF16 = True
USE_FP16 = False
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 3
GRADIENT_CHECKPOINTING = True
PAD_VALUE = 0.0
MAX_LENGTH = 8192

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def initialize_wandb():
    """Initialize Weights and Biases for tracking experiments."""
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project=WANDB_PROJECT_NAME,
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
        }
    )


class CustomDataset(Dataset):
    """Custom dataset class for handling audio-text data."""

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data
        audio_unit = np.array(entry["unit"][idx])
        x_vector = entry["x-vector"][idx]
        text_with_pad = entry["text_with_pad"][idx]

        padding_token = 2048
        bos_token_id = 2049  # Start of Sequence token ID
        eos_token_id = 2030  # End of Sequence token ID

        audio_unit = np.hstack((audio_unit, np.zeros((audio_unit.shape[0], 1), dtype=int)))
        for i in range(1, audio_unit.shape[0]):
            audio_unit[i, 1:] = audio_unit[i, :-1]
            audio_unit[i, 0] = padding_token

        matrix_with_bos = np.hstack((np.full((audio_unit.shape[0], 1), bos_token_id), audio_unit))
        matrix_with_bos_eos = np.hstack((matrix_with_bos, np.full((matrix_with_bos.shape[0], 1), eos_token_id)))
        input_audio_unit = matrix_with_bos_eos[:, :-1]
        target_audio_unit = matrix_with_bos_eos[:, 1:]

        return {
            "input_values": torch.tensor(input_audio_unit),
            "codec_label": torch.tensor(target_audio_unit),
            "speaker_emb": torch.tensor(x_vector),
            "tts_text": self.tokenizer(text_with_pad, add_special_tokens=False, return_tensors="pt")[
                "input_ids"].squeeze(0)
        }


class CustomDataCollator:
    """Custom data collator for batching audio and text inputs."""

    def __init__(self, text_pad_value, audio_pad_value=PAD_VALUE):
        self.text_pad_value = text_pad_value
        self.audio_pad_value = audio_pad_value

    def __call__(self, batch):
        # input_values = torch.nn.utils.rnn.pad_sequence(
        #     [item["input_values"] for item in batch],
        #     batch_first=True,
        #     padding_value=self.audio_pad_value
        # )
        # labels = torch.nn.utils.rnn.pad_sequence(
        #     [item["tts_texts"] for item in batch],
        #     batch_first=True,
        #     padding_value=self.text_pad_value
        # )
        return {
            "input_values": torch.cat([item["input_values"] for item in batch]),
            "speaker_emb": torch.cat([item["speaker_emb"] for item in batch]),
            "tts_text": torch.cat([item["tts_text"] for item in batch]),
            "asr_texts": torch.cat([item["asr_texts"] for item in batch]),
        }


def compute_metrics(pred):
    """Compute loss as a metric."""
    pred_logits = pred.predictions
    labels = pred.label_ids
    loss_fn = torch.nn.CrossEntropyLoss()
    return {"loss": loss_fn(torch.tensor(pred_logits), torch.tensor(labels)).item()}


def main():
    # Initialize WandB if in main process
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        initialize_wandb()

    # Load model and tokenizer
    config = MMLMTTSConfig(lm_model_name=LM_MODEL_NAME)
    model = MMLMTTS(config)
    tokenizer = model.tokenizer
    logger.info("Model and tokenizer loaded.")

    # Load dataset
    data = DATASET
    logger.info(f"Loaded {len(data)} samples from dataset.")
    data = data.filter(lambda x: len(x["unit"]) <= MAX_LENGTH)
    logger.info(f"Filtered dataset to {len(data)} samples.")

    # Split dataset
    data = data.train_test_split(test_size=TRAIN_TEST_SPLIT_RATIO, seed=42)
    train_dataset = CustomDataset(data['train'], tokenizer)
    eval_dataset = CustomDataset(data['test'], tokenizer)

    # Data collator
    data_collator = CustomDataCollator(tokenizer.pad_token_id)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        save_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_accumulation_steps=10,
        learning_rate=LEARNING_RATE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        bf16=USE_BF16,
        fp16=USE_FP16,
        report_to="wandb",
        run_name=f"{WANDB_PROJECT_NAME}-training",
        load_best_model_at_end=False,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        label_names=["tts_label", "codec_label"],
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train and evaluate model
    trainer.train()
    trainer.evaluate()

    # Save model
    trainer.save_model(MODEL_SAVE_PATH)
    logger.info(f"Model and tokenizer saved to '{MODEL_SAVE_PATH}'.")

    # Finalize WandB
    wandb.finish()


if __name__ == "__main__":
    main()
