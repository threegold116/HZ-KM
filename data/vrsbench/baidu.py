import requests
import random
import json
from hashlib import md5
import time
# Set your own appid/appkey.
appid = '20231022001855511'
appkey = 'WCXg7pcWKmOwSsoMJlWn'
 
# For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
from_lang = 'zh'
to_lang =  'en'
# from_lang = 'en'
# to_lang =  'zh'
 
endpoint = 'http://api.fanyi.baidu.com'
path = '/api/trans/vip/translate'
url = endpoint + path
url="https://fanyi-api.baidu.com/api/trans/vip/translate"
 
query = '你好！世界'
cache={}
# print("query:{}".format(query))
# Generate salt and sign


def is_chinese(input_string):
    for char in input_string:
        if  '\u4e00' <= char and char <= '\u9fff':
            # print(input_string)
            print("\nis_chinese!-------")
            return True
    return False
def is_english(input_string):
    for char in input_string:
        if  '\u4e00' > char or char > '\u9fff':
            # print(input_string)
            print("\nis_english!-------")
            return True
    return False
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()

def remove_english(text):
    filtered_text = ""
    for char in text:
        if ord(char)>=128:
            filtered_text += char   
    return filtered_text


def baidu_api(query,from_lang='zh',to_lang='en'):
    print(query)
    print("FROM {} TO {}".format(from_lang,to_lang))

    if from_lang=='zh':
        query=remove_english(query)
        print(f"after_remove:{query}")
    if query in cache.keys():
        print("CACHE!!!")
        print(f"after retans:{cache[query]}")
        return cache[query]
    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)
 
    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}
 
    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()
    print(f"after retans:{result['trans_result'][0]['dst']}")
    # if not is_chinese(result['trans_result'][0]['dst']):
    if query not in cache.keys() and not is_english(result['trans_result'][0]['dst']):
        cache[query]=result['trans_result'][0]['dst']
    # Show response
    #print(json.dumps(result, indent=4, ensure_ascii=False))
    time.sleep(1)
    return result["trans_result"][0]['dst']
 

baidu_api("hello","en","zh")


