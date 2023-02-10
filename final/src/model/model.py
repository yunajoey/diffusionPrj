# Hugging face에서 사전학습된 모델과 tokenizer로드하기 
# 나중에 argparse로 명령어로 model Class를 쳤을 때 그 클래스에 해당되는 model_name이 나오게 하기 
# 출처: https://github.com/Inju0716/ko-en-neural-machine-translation/blob/main/Huggingface_Pretrained_Model.ipynb


import argparse 
import logging  
import numpy as np
import torch

from transformers import (
        AutoModelForSeq2SeqLM, 
        AutoTokenizer,
)   

MODEL_CLASSES = {  
    "Auto" : (AutoTokenizer,  AutoModelForSeq2SeqLM)  
}  


MODEL_NAMES_LIST = {   
      "facebook": "facebook/mbart-large-cc25",  
      "t5": "t5-small",
}   
      
model_class, tokenizer_class = MODEL_CLASSES['Auto']   

tokenizer = model_class.from_pretrained("t5-small")  
model = tokenizer_class.from_pretrained("t5-small")





