# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("voidful/SmolLM2-360M-Instruct-ASR")
model = AutoModelForCausalLM.from_pretrained("voidful/SmolLM2-360M-Instruct-ASR",ignore_mismatched_sizes=True)
print(len(tokenizer))
add_tokens = ([f"tts_tok_{u}" for u in range(1024 * 10)] +
              ["[PAD]", "[END_PAD]"])
num_added_toks = tokenizer.add_tokens(add_tokens)
print(len(tokenizer))
model.resize_token_embeddings(len(tokenizer))

tokenizer.push_to_hub("voidful/SmolLM2-360M-Instruct-TTS")
model.push_to_hub("voidful/SmolLM2-360M-Instruct-TTS")
