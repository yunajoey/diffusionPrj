# test 데이터 셋으로 blue_score로 평가하기 
# 출처:  https://github.com/Inju0716/ko-en-neural-machine-translation/blob/main/Huggingface_Pretrained_Model.ipynb
import torch 
from train import * 
from torchtext.data.metrics import bleu_score
from torchmetrics.functional import bleu_score
from transformers import (
        AutoModelForSeq2SeqLM, 
        AutoTokenizer,
)   


