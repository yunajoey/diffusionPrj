# https://github.com/Inju0716/ko-en-neural-machine-translation/blob/main/Huggingface_Pretrained_Model.ipynb
from src.data.utils import  CustomDataset, split_dataSet
from src.model.model import model_print
from transformers import Trainer, TrainingArguments  
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import DatasetDict, load_from_disk
import os 

tokenizer, model = model_print()
data_path = "./src/data"
ko_dir = os.path.join(data_path,"ko_shuffle.txt")
en_dir = os.path.join(data_path, "en_shuffle.txt")  

with open(ko_dir, 'r', encoding='UTF-8') as f:
  ko_text = f.readlines()
with open(en_dir, 'r', encoding='UTF-8') as f:
  en_text = f.readlines()    

# dataSet = CustomDataset(ko_text, en_text, tokenizer) 

# train_data, valid_data, test_data = split_dataSet(dataSet, 0.7, 0.2) 

 
dataSet = CustomDataset(ko_text, en_text, tokenizer)  
train_data, valid_data, test_data = split_dataSet(dataSet, 0.7, 0.2)  
dataset = DatasetDict({
    "train": train_data,
    "validation": valid_data,
    "test": test_data,
})  

try:
  if not os.path.exists('./output'):
      os.makedirs('./output')   
except:
   pass  

# fine-tuning the model   
training_args = TrainingArguments(
    output_dir='./checkpoint', 
    num_train_epochs=2,
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    warmup_steps=500,
    save_steps=100,
    save_total_limit=5,
    do_eval=True,
    save_strategy='epoch',       
    # predict_with_generate=True,  seq2seqtrainer 가 매개변수로 갖는것
    # fp16=True,
    # load_best_model_at_end= True,
    weight_decay=0.01     
)    

trainer = Trainer(
    model=model,
    tokenizer = tokenizer, 
    args=training_args,
    train_dataset=train_data,
    eval_dataset=valid_data,    
)      

if __name__ == "__main__":     
    trainer.train()   
    trainer.save_model('./output')