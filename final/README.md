
# 프로젝트 폴더 구조 

```
final
├─ .gitignore
├─ checkpoint
├─ evalute.py
├─ main.py
├─ output
├─ README.md
├─ requirements.txt
├─ src
│  ├─ API
│  │  └─ api.py
│  ├─ data
│  │  ├─ dataSet.json
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

## 3. evaluate.py 
 - 트레이닝 시킨 모델을 blue Score로 평가하는 .py   
  <hr>   
 
## 4. api.py
 - 사용할 API(naver, google, python 내장)가 있는 .py   
  <hr>     
 
## 5. api.py
 - 사용할 API(naver, google, python 내장)가 있는 .py 
  
## 6. main.py
 - dffusion_model에 들어갈 3가지 타입의 text 형태(수정하지않은 번역 text, 태깅작업한 번역 text, magic_word를 넣은 번역 text)를 json 파일로 ouput하는 .py
 
 ## 7. output폴더 
 - 훈련한 model를 저장하는 폴더
 
 ## 8. checkpoint 폴더 
 - checkpoint가 기록이 되는 폴더
 

