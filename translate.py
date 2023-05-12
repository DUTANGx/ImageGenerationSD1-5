import json
import requests
import unicodedata

def contains_chinese_characters(input_string):
    for char in input_string:
        if "CJK" in unicodedata.name(char):
            return True
    return False

def translate(word):
    if contains_chinese_characters(word) == False:
        return word
    url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=null'
    key = {
        'type': "AUTO",
        'i': word,
        "doctype": "json",
        "version": "2.1",
        "keyfrom": "fanyi.web",
        "ue": "UTF-8",
        "action": "FY_BY_CLICKBUTTON",
        "typoResult": "true"
    }
    response = requests.post(url, data=key)
    if response.status_code == 200:
        result = json.loads(response.text)
        return result['translateResult'][0][0]['tgt']
    else:
        # 翻译模块失败，prompt请用英文输入
        return None

if __name__ == '__main__':
    word = '一个可以释放闪电的小女孩'
    translation = translate(word)
    word = 'most recent call last'
    translation = translate(word)
    print(translation)