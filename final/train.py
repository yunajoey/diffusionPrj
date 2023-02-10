# 출처: https://bo-10000.tistory.com/154
# 출처: https://flonelin.wordpress.com/2021/11/29/transformers-trainer%EC%97%90%EC%84%9C-earlys-stopping-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0/

from src.data.utils import  *
from src.model.model import * 
from transformers import Trainer, TrainingArguments
# from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchtext.data.metrics import bleu_score


dataSet = CustomDataset(ko_text, en_text, tokenizer) 

train_data, valid_data, test_data = split_dataSet(dataSet, 0.5, 0.3)

try:
  if not os.path.exists('./output'):
      os.makedirs('./output')   
except:
   pass  

# fine-tuning the model  

training_args = TrainingArguments(
    output_dir='./checkpoint', # checkpoint는 checkpoint 폴더 아래에다가 
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
    # 트레이닝 시키고 
    trainer.train()
    # 모델을 저장한다
    trainer.save_model('./output')   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    