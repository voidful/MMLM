from mmlm.model import MMLM
from mmlm.utility import MMLMUtility
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, AutoModel

lm_model = AutoModelForCausalLM.from_pretrained('voidful/phi-1_5_chat_128k')
lm_tokenizer = AutoTokenizer.from_pretrained('voidful/phi-1_5_chat_128k')
audio_model = AutoModel.from_pretrained('ntu-spml/distilhubert')

mmlm = MMLM('voidful/phi-1_5_chat_128k', lm_model=lm_model, lm_tokenizer=lm_tokenizer, audio_config=8)
mmlu = MMLMUtility(mmlm)

dataset = load_dataset("voidful/cv_13_tw_speech_tokenizer_asr")

tokenized_datasets = dataset.map(mmlu.tokenize_function, batched=False)

dc = mmlu.MMLMDataCollator(mmlm.tokenizer)

mmlm.tokenizer.pad_token = mmlm.tokenizer.eos_token
training_args = TrainingArguments(
    output_dir='./results_asr',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    logging_steps=1,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initialize the Trainer
trainer = Trainer(
    model=mmlm,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=lm_tokenizer,
    data_collator=dc
)

trainer.train()
