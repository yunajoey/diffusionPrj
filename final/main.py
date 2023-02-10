
from src.API.api import google_translate, papago_api, trans_lib
import pandas as pd
import nltk
import torch
from nltk.data import load
from nltk.tokenize import word_tokenize  
import random  
from PIL import Image, ImageDraw
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers import UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# from src.data.utils import  *
# from src.model.model import * 
# from train import * 
# from transformers import Trainer, TrainingArguments

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = AutoModelForSeq2SeqLM.from_pretrained('./checkpoint/checkpoint-12')  
# tokenizer = AutoTokenizer.from_pretrained('./checkpoint/checkpoint-12')
# model.to(device)
# inputs = tokenizer('안녕하세요', return_tensors="pt")
# inputs.to(device)
# output = model.generate(inputs["input_ids"])
# tokenizer.decode(output[0].tolist())



magic_keywords = ['hdr, uhd, 64k', 'highly detailed', 'studio lighting', 'professional', 'trending on artstation', 
                  'unreal engine', 'vivid Colors', 'bokeh', 'sketch of', 'painting of']


def magicword_text(translate_text:list) -> list:
    num = 2
    selected_magic_word = random.sample(magic_keywords,num)
    return translate_text + "," + "".join(selected_magic_word)
    
 
def cleaned_text(translated_text:str, *args): 
    """magic_word_keyword는 사용자 마음대로 넣을수도 있고 아닐수도 있고!"""
    stop_words_list = list()  
    with open('stopwords.txt', 'r') as f:
        i = [line.strip() for line in f.readlines()]
        stop_words_list.append(i)   
    stop_words_list = stop_words_list[0]
    
    tokend_list = word_tokenize(translated_text)
    tokend_tagging_list = nltk.pos_tag(tokend_list)
    wanted_tagging_list = ["NN", "NNS", "NNP", "NNPS", "VB", "VBG", "VBD", "VBN", "VBP", "VBZ", "JJ", "PDT", "CD"]  
    result_dic = dict()
    for tokened, tag in tokend_tagging_list:
        if tag in wanted_tagging_list:
            if tag not in result_dic: 
                result_dic[tag] = [tokened]
            else:
                result_dic[tag].append(tokened)            
    prompt_text = " ".join(list(" ".join(v) for v in result_dic.values()))
    magic_words = ",".join(list(args))
    return prompt_text +" , "+magic_words



def different_img_result(original_text, customed_text, without_magic_word_text, n_pics):
  original_prompt = [original_text] * n_pics 
  customed_prompt = [customed_text] * n_pics 
  without_magic_prompt = [without_magic_word_text] * n_pics
  with autocast(device):
     img1 = pipe(original_prompt, num_inference_steps=50).images
     img2 = pipe(customed_prompt, num_inference_steps=50).images
     img3 = pipe(without_magic_prompt, num_inference_steps=50).images
  return img1, img2, img3


def image_grid(imgs, rows, cols):
  assert len(imgs) == rows*cols
  w, h = imgs[0].size
  grid = Image.new('RGB', size=(cols*w, rows*h))
  grid_w, grid_h = grid.size

  for i, img in enumerate(imgs):
      grid.paste(img, box=(i%cols*w, i//cols*h))
  return grid

def get_concat_h_multi_resize(original, customed, magic_word, resample=Image.BICUBIC):
    # original, customed, magic_word = different_img_result(original_text, customed_text, magic_word_text, 1)

    img1 = image_grid(original, rows=1, cols=1)  
    img2 = image_grid(customed, rows=1, cols=1)
    img3 = image_grid(magic_word , rows=1, cols=1)   

    im_list = [img1, img2, img3]     
    min_height = min(im.height for im in im_list)
    im_list_resize = [im.resize((int(im.width * min_height / im.height), min_height),resample=resample)
                      for im in im_list]
    total_width = sum(im.width for im in im_list_resize)
    dst = Image.new('RGB', (total_width, min_height))
    pos_x = 0
    for im in im_list_resize:
        dst.paste(im, (pos_x, 0))
        pos_x += im.width
    return dst




if __name__ == "__main__":   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', revision='fp16',torch_dtype=torch.float16, use_auth_token=True)
    pipe = pipe.to(device)


    df = pd.DataFrame(["버거킹을 먹는 여자아이",
                        "두명의 남자아이이가 놀이터에서 놀고 있는 그림",
                        "목도리와 모자를 쓴 눈사람",
                        "해가 떠오르는 바다",
                        "놀이터에서 놀고 있는 소녀와 소년",
                        "커피와 초콜렛을 먹는 여자",
                        "스타벅스에서 커피를 주문하는 여자",
                        "네 감사합니다.",
                        "유명한 레스토랑에서 브런치를 먹는 사람들.",
                        "놀이기구를 타려고 줄서고 있는 커플"]
                        , columns = ['Kor'])
    
    original_translated_list = df['Kor'].apply(google_translate).tolist() 
    customed_text_list = list(map(cleaned_text, original_translated_list))
    magic_word_text_list = list(map(magicword_text, original_translated_list))
   
 
