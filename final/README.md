
```
final
├─ .gitignore
├─ checkpoint
├─ eval.py
├─ main.py
├─ output
├─ predict.py
├─ README.md
├─ requirements.txt
├─ show_img.py
├─ src
│  ├─ API
│  │  └─ api.py
│  ├─ data
│  │  ├─ sample_data.json
│  │  └─ utils.py
│  └─ model
│     └─ model.py
├─ stopwords.txt
└─ train.py

```

## 폴더 & 파일 설명 ## 
1. src폴더  
1-1) API폴더 
- api.py => 사용할 api가 있는 .py

1-2) data폴더 
- 모델을 training시킬 data가 있는 폴더  

1-3) model폴더 
- model.py => 사용할 model이 있는 .py

2. output 폴더 
- 모델이 training후 들어갈 폴더 

3. predict.py 
- training한 model로 따로 문장을 predict하고 싶을때 사용하는 .py 

4. main.py 
- model 또는 api 사용할 시, 현재 정의되어 있는 데이터들의 
번역을 하게 되는 .py

5. eval.py 
- model의 blue_scroe를 평가하는 .py 


## 사용법 ## 
<b> 1. 모델을 사용할 경우 <b>

Step1.
- python train.py   
- Model Name >> [사용할모델이름]

Step2.
- python main.py --model [사용할모델이름]

Step3.(option)
- python predict.py --text [번역하고 싶은 문장]

<b> 2. API를 사용할 경우 <b>
- python main.py (default값이  google API)
- python main.py --api [사용할API]

-------------------------------------------------------
사용가능한 모델 LIST들
- QuoA

