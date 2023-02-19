
from PIL import Image, ImageDraw
from diffusers import StableDiffusionPipeline, AutoencoderKL
import torch 
import json

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

pipe = StableDiffusionPipeline.from_pretrained(
    'CompVis/stable-diffusion-v1-4', revision='fp16',
    torch_dtype=torch.float16, use_auth_token=True)
pipe = pipe.to(device)



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



f = open('.prompt_data.json')
data = json.load(f)
prompt_text = json.loads(data['data'])
original, custom, magic = [[i['original'], i['custom'], i['magic']] for i in prompt_text][0]  # 0 (숫자 바꺼 가면서 하면됨)

original, customed, magic_word = different_img_result(original, custom, magic, 1)

get_concat_h_multi_resize(original, customed, magic_word)