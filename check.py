
import json
import os
import sys
import re
import urllib.request
from translate import Translator
from typing import List, Dict



client_id = "H4q5ybS5cfAmEl1hDUqC" # 개발자센터에서 발급받은 Client ID 값
client_secret = "w5KyEQhM2l" # 개발자센터에서 발급받은 Client Secret


def cleaned_text(strs):
    p = re.search(r'(?<=\"translatedText").+', strs)
    return p.group()

def papago_api(input_text, cliend_id, client_secret):
    encText = urllib.parse.quote(input_text)
    data = "source=ko&target=en&text=" + encText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode() 
    if(rescode==200):
        response_body = response.read()
        response = response_body.decode('utf-8') 
        result =  response.split('"engineType"')
        result = result[0]  
        text = cleaned_text(result)
        text = text.replace(':',"").replace(',',"").replace('"',"").replace('.',"")
    return text

dataSet_result = dict()  # 기존의 한국어, 영어 번역 셋
api_result = dict()  # 파파고 api를 이용하여 번역한 한국어, 영어 번역 셋
with open("sample.json", "r", encoding='UTF-8') as file: 
    st_python = json.load(file)
    data : List[Dict] = st_python['data']
    for d in data:
        d: Dict[str, str]
        kor, eng = d['ko_original'].replace('>',""), d['en'].replace('>',"")         
        dataSet_result[kor] = eng 
        api_result[kor] = papago_api(kor,client_id, client_secret)

# 저장하기 
with open("api_result.json", "w", encoding='utf-8-sig') as f:
    json.dump(api_result, f, ensure_ascii=False, indent=4)
    
with open("dataSet_result.json", "w", encoding='utf-8-sig') as f:
    json.dump(dataSet_result, f, ensure_ascii=False, indent=4)

# with open("dataSet_result.json", "w", encoding='utf-8-sig') as f:
#     f.write(json.dumps(dataSet_result, ensure_ascii=False))