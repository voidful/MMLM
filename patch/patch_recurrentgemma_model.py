import nlp2
import torch
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/recurrentgemma-2b")
pretrained = AutoModelForCausalLM.from_pretrained("google/recurrentgemma-2b")

print(tokenizer.tokenize(
    "Hello world CAUDIO_TAG_0 CVISUAL_TAG_0 a_tok_1 v_tok_1 <|im_start|> <|im_start|> <|im_end|> <|endoftext|> [END] [Decomposition] [Queries]"))

add_tokens = ([i[0] for i in nlp2.read_csv('zh_tf_list.csv')[:10000]] +
              [f"v_tok_{u}" for u in range(1024 * 10)] +
              [f"a_tok_{u}" for u in range(1024 * 10)] +
              [f"CAUDIO_TAG_{u}" for u in range(100)] +
              [f"CVISUAL_TAG_{u}" for u in range(100)] +
              ["<|im_start|>", "<|im_end|>", "<|endoftext|>"] +
              ["[END]", "[Decomposition]", "[Queries]"])

origin_vocab_size = tokenizer.vocab_size
print("===ADD TOKEN===")
num_added_toks = tokenizer.add_tokens(add_tokens)
num_added_toks = len(tokenizer)-origin_vocab_size
print('We have added', num_added_toks, 'tokens',len(tokenizer)-origin_vocab_size)
print(origin_vocab_size, num_added_toks, len(tokenizer))
print(tokenizer.tokenize(
    "Hello world CAUDIO_TAG_0 CVISUAL_TAG_0 a_tok_1 v_tok_1 <|im_start|> <|im_start|> <|im_end|> <|endoftext|>[END][Decomposition][Queries]"))

print("===============")

pretrained.resize_token_embeddings(len(tokenizer))
input_embedding = pretrained.get_input_embeddings()
state_dict_weight = input_embedding.state_dict()['weight']
random_indices = torch.randperm(origin_vocab_size)[:num_added_toks]
new_tokens_weights = state_dict_weight[random_indices]
print(origin_vocab_size,origin_vocab_size + num_added_toks,new_tokens_weights.shape)
state_dict_weight[origin_vocab_size:origin_vocab_size + num_added_toks] = copy.copy(new_tokens_weights)
pretrained.set_input_embeddings(input_embedding)
print("===============")

pretrained.config.rotary_emb_base = 50000.0
# pretrained.config.max_position_embeddings = 2048
# pretrained.config.rope_scaling = {"type": "dynamic", "factor": 16.00}

tokenizer.push_to_hub("voidful/recurrentgemma-2b-base")
pretrained.push_to_hub("voidful/recurrentgemma-2b-base")
