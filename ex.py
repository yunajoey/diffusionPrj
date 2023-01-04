import os
import sys
import re
import urllib.request
from translate import Translator


def cleaned_text(strs):
    p = re.search(r'(?<=\"translatedText").+', strs)
    return p.group()

input_words = "아침에는커피점심에는빵"

client_id = "_rbq2co1IRIcXuH6u1yk" # 개발자센터에서 발급받은 Client ID 값
client_secret = "9CoOVn797z" # 개발자센터에서 발급받은 Client Secret 값
encText = urllib.parse.quote(input_words)
data = "source=ko&target=en&text=" + encText
url = "https://openapi.naver.com/v1/papago/n2mt"
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request, data=data.encode("utf-8"))
rescode = response.getcode()
dic = dict()
if(rescode==200):
    response_body = response.read()
    response = response_body.decode('utf-8') 
    result =  response.split('"engineType"')
    result = result[0]  
    text = cleaned_text(result)
    text = text.replace(':',"").replace(',',"").replace('"',"").replace('.',"")
    dic["translatedText"] = text           
        
else:
    print("Error Code:" + rescode)  
 
# print(dic) {'translatedText': 'Coffee for breakfast bread for lunch'}

translator= Translator(to_lang="ko")
Eng_to_kor = translator.translate(dic['translatedText'])