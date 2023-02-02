### text-to-img를 구현하는 모델 repository

#### 1.papago.py >> Naver API로 번역이 괜찮게 되는지 확인하는 .py

#### 2.G_Model.py >> diffusion이미지를 잘 표현하도록 text-generation하는 pretrained된 모델인 Gustovosta 모델로 text-generation하는 .py

#### 3.text_generation.py >> text_generation을 할 수 있는 여러가지 모델들을 사용해 볼 수 있는 .py 
- how to use it?!
- model type은 model_classes에 써져 있음(허깅페이스에서 가져와서 add_up해도 된다)
- model_name에는 허깅페이스에서 model_type에 해당하는 mdoel_name중 사용하고 싶은 것을 치면은 된다 
예를 들어) 
<p> python test-generation.py --model_type=gustava --model_name=Gustavosta  MagicPrompt-Stable-Diffusion </p> 
<p> python test-generation.py --model_type=gpt2 --model_name=gpt2 </p>
<p> python test-generation.py --model_type=bloom --model_name=bloom-17-b  </p>


