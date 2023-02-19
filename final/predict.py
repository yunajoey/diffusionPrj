from torchtext.data.metrics import bleu_score
import torch 
import os


from transformers import (
        AutoModelForSeq2SeqLM, 
        AutoTokenizer,
)   



device = torch.device('cude:0' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSeq2SeqLM.from_pretrained('./checkpoint/checkpoint-16')
tokenizer = AutoTokenizer.from_pretrained('./checkpoint/checkpoint-16')
    


sen_list = ['오늘도좋은하루보내세요']

translate_input = tokenizer.prepare_seq2seq_batch(sen_list, return_tensors="pt")
translate_input.to(device)
translated = model.generate(**translate_input)
trg_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
print(trg_text)
