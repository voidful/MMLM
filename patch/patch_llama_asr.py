import torch
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor

tokenizer = AutoTokenizer.from_pretrained("voidful/Llama-3.2-8B-Instruct")
pretrained = AutoModelForCausalLM.from_pretrained("voidful/Llama-3.2-8B-Instruct")
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")

whisper_dict = processor.tokenizer.get_vocab()
llama_dict = tokenizer.get_vocab()

intersection = {i: llama_dict[i]
                for i in set(whisper_dict.keys()).intersection(set(llama_dict.keys()))}
symmetric_difference = {i: whisper_dict[i]
                for i in set(intersection.keys()).symmetric_difference(set(whisper_dict.keys()))}

# asset len(intersection)+len(symmetric_difference) should be equal to len(whisper_dict)
assert len(intersection)+len(symmetric_difference) == len(whisper_dict)

# backup embedding layer
state_dict_weight_clone = pretrained.get_input_embeddings().state_dict()['weight'].clone()
# create new inputting embedding layer with size equal to len(whisper_dict)
pretrained.resize_token_embeddings(len(whisper_dict))
input_embedding = pretrained.get_input_embeddings()
state_dict_weight = input_embedding.state_dict()['weight']
for i in intersection:    state_dict_weight[whisper_dict[i]] = state_dict_weight_clone[llama_dict[i]]
pretrained.set_input_embeddings(input_embedding)

# check if the new embedding layer is correctly set
assert torch.equal(pretrained.get_input_embeddings().state_dict()['weight'], state_dict_weight)

add_tokens = (["[PAD]", "[END_PAD]"]+
                tokenizer.all_special_tokens)
num_added_toks = processor.tokenizer.add_tokens(add_tokens)
pretrained.resize_token_embeddings(len(processor.tokenizer))

processor.tokenizer.push_to_hub("voidful/Llama-3.2-8B-ASR")
pretrained.push_to_hub("voidful/Llama-3.2-8B-ASR")
