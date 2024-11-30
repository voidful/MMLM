import os
import json
import logging
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import wandb
from mmlm.utility import load_audio_to_tensor
from mmlm.model_asr import MMLMASR, MMLMASRConfig

# ========================
# Global Configuration
# ========================
WANDB_PROJECT_NAME = "mmlm-asr"
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
DATA_PATH = os.environ.get("DATA_PATH")
LM_MODEL_NAME = "voidful/SmolLM2-360M-Instruct-ASR"
OUTPUT_DIR = "./mmlm-asr-training"
MODEL_SAVE_PATH = "./mmlm-asr-model"
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
        entry = self.data[idx]
        audio_path = entry["audio_path"]
        text = entry["text_with_pad"]

        # Load and preprocess audio
        audio_tensor = load_audio_to_tensor(audio_path).squeeze()
        if audio_tensor.nelement() == 0:
            raise ValueError(f"Empty audio tensor at index {idx}")

        # Tokenize text
        text_inputs = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        if text_inputs["input_ids"].nelement() == 0:
            raise ValueError(f"Empty text input at index {idx}")

        return {
            "input_values": audio_tensor,
            "asr_texts": text_inputs["input_ids"].squeeze(0)
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
        #     [item["asr_texts"] for item in batch],
        #     batch_first=True,
        #     padding_value=self.text_pad_value
        # )
        return {
            "input_values": torch.cat([item["input_values"] for item in batch]),
            "asr_texts": torch.cat([item["asr_texts"] for item in batch]),
        }


def load_data(data_path):
    """Load dataset from a JSONL file."""
    data = []
    with open(data_path, "r", encoding="utf-8") as file:
        for line in file:
            entry = json.loads(line)
            if len(entry["text_with_pad"].split("[PAD]")) < MAX_LENGTH and os.path.exists(entry["audio_path"]):
                data.append(entry)
    return data


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
    config = MMLMASRConfig(lm_model_name=LM_MODEL_NAME)
    model = MMLMASR(config)
    tokenizer = model.tokenizer
    logger.info("Model and tokenizer loaded.")

    # Load dataset
    data = load_data(DATA_PATH)
    logger.info(f"Loaded {len(data)} samples from dataset.")

    # Split dataset
    train_data, eval_data = train_test_split(data, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=42)
    train_dataset = CustomDataset(train_data, tokenizer)
    eval_dataset = CustomDataset(eval_data, tokenizer)

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
        learning_rate=LEARNING_RATE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        bf16=USE_BF16,
        fp16=USE_FP16,
        report_to="wandb",
        run_name=f"{WANDB_PROJECT_NAME}-training",
        load_best_model_at_end=False,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        label_names=["asr_labels"],
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
