
# 프로젝트 폴더 구조 

```
diffusion
├─ .gitignore
├─ checkpoint
├─ evalute.py
├─ main.py
├─ output
│  ├─ config.json
│  ├─ pytorch_model.bin
│  ├─ special_tokens_map.json
│  ├─ spiece.model
│  ├─ tokenizer.json
│  ├─ tokenizer_config.json
│  └─ training_args.bin
├─ requirements.txt
├─ src
│  ├─ API
│  │  └─ api.py
│  ├─ data
│  │  ├─ dataSet.json
│  │  ├─ en_shuffle.txt
│  │  ├─ ko_shuffle.txt
│  │  ├─ sample_data.json
│  │  └─ utils.py
│  └─ model
│     └─ model.py
├─ stopwords.txt
└─ train.py

```

### 각 폴더 설명<br>
## 1. src폴더 
 - 사용할 API, model, data가 있는 폴더이다<br>
 1-1) API폴더에는 naver, google, python 내장 번역 API가 있다<br>
 1-2) model 폴더에는 hugging face에서 translate 할 수 있는 모델이 있는 폴더가 있다<br>
 1-3) data 폴더에는 sample_data.json(데이터갯수: 약 100개), dataSet.json(데이터갯수: 약 3만개)<br>
 <hr>
 ## 2. train.py 
  - model를 트레이닝 시키는 .py 
 <hr>
 ## 3. evaluate.py 
  - 트레이닝 시킨 모델을 blue Score로 평가하는 .py 
 <hr>
 ## 4. main.py 
  - 사용할 API 또는 모델로 번역(한국어-> 영어) 와  diffusion model이 잘 인식 할 수 있는 prompt message로 처리 하는 .py 
 <hr>
 
