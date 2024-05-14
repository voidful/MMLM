from transformers import AutoModelForCausalLM, AutoTokenizer
from mmlm.model import MMLM
from datasets import load_dataset

lm_model = AutoModelForCausalLM.from_pretrained('voidful/pythia-160m')
lm_tokenizer = AutoTokenizer.from_pretrained('voidful/pythia-160m')
# audio_model =  AutoModel.from_pretrained('ntu-spml/distilhubert')
ds = load_dataset("distil-whisper/librispeech_asr_dummy", split='validation')
dataset = load_dataset("Codec-SUPERB/librispeech_asr_dummy_extract_unit")

# merge two dataset
codec_layer = 8
input_datas = []
d = dataset['speech_tokenizer_16k']
for b in range(len(d)):
    encodec_input = ""
    for i in range(codec_layer):
        encodec_input = "".join([f"a_tok_{u + i * 1024}" for u in d[b]['unit'][i]])
    input_datas.append([encodec_input, ds[b]['text']])

# init model
mmlm = MMLM('voidful/pythia-160m',lm_model=lm_model,lm_tokenizer=lm_tokenizer,audio_config=8)

# prediction before training
test_input_text = "transcribe the audio:"+input_datas[9][0]+"\nTranscription:"
test_input_ids = mmlm.tokenizer(test_input_text, return_tensors="pt").input_ids
outputs = mmlm.generate(test_input_ids, max_length=100)
print(mmlm.tokenizer.batch_decode(outputs))

# training loss
input_text = "transcribe the audio:"+input_datas[9][0]+"\nTranscription:"+input_datas[9][1]
target_text = input_datas[9][1] + mmlm.tokenizer.eos_token
input_ids = mmlm.tokenizer(input_text, return_tensors="pt").input_ids
target_ids = mmlm.tokenizer(target_text, return_tensors="pt").input_ids
loss = mmlm.forward(input_ids,labels=target_ids).loss
print(loss)

# training loop
import torch.optim as optim
for _ in range(100):
    optimizer = optim.Adam(mmlm.parameters(), lr=0.0001)
    loss = mmlm.forward(input_ids,labels=target_ids).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(loss)

# prediction after training
test_input_text = "transcribe the audio:"+input_datas[9][0]+"\nTranscription:"
test_input_ids = mmlm.tokenizer(test_input_text, return_tensors="pt").input_ids
outputs = mmlm.generate(test_input_ids, max_length=100)
print(mmlm.tokenizer.batch_decode(outputs))
