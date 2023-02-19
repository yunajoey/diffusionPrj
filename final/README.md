
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

