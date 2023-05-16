import json
import requests
import unicodedata
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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


def model_translate(word):
    if contains_chinese_characters(word) == False:
        return word
    
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

    # Tokenize the text
    batch = tokenizer.prepare_seq2seq_batch(src_texts=[word], return_tensors='pt', max_length=512)

    # Make sure that the tokenized text does not exceed the maximum
    # allowed size of 512
    batch["input_ids"] = batch["input_ids"][:, :512]
    batch["attention_mask"] = batch["attention_mask"][:, :512]

    # Perform the translation and decode the output
    translation = model.generate(**batch)
    result = tokenizer.batch_decode(translation, skip_special_tokens=True)

    return result[0]


if __name__ == '__main__':
    word = '一个可以释放闪电的小女孩'
    translation = translate(word)
    word = 'most recent call last'
    translation = translate(word)
    print(translation)