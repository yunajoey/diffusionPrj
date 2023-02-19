from torchtext.data.metrics import bleu_score
import torch 
import os


from transformers import (
        AutoModelForSeq2SeqLM, 
        AutoTokenizer,
)   

data_path = "./src/data"
ko_dir = os.path.join(data_path,"ko_shuffle2.txt")
en_dir = os.path.join(data_path, "en_shuffle2.txt")  


with open(ko_dir, 'r', encoding='UTF-8') as f:
  ko_text = f.readlines()
with open(en_dir, 'r', encoding='UTF-8') as f:
  en_text = f.readlines()    

device = torch.device('cude:0' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSeq2SeqLM.from_pretrained('./checkpoint/checkpoint-16')
tokenizer = AutoTokenizer.from_pretrained('./checkpoint/checkpoint-16')
    
for i in range(0,10,2):
  model_generate_input= tokenizer.prepare_seq2seq_batch(ko_text[i:i+2], return_tensors="pt").to(device)
  translated = model.generate(**model_generate_input)
  tgt_text_batch = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
  print(tgt_text_batch)