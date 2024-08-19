
translate_map = {
    'one': '1', 
    'two': '2', 
    'three': '3', 
    'four': '4',
    'five': '5', 
    'six': '6', 
    'seven': '7', 
    'eight': '8',
    'nine': '9', 
    'ten': '10',
    'eleven': '11',
    'twelve': '12',
    'thirteen': '13',
    'fourteen': '14',
    'fifteen': '15',
    'sixteen': '16',
    'seventeen': '17',
    'eighteen': '18',
    'nineteen': '19',
    'twenty': '20',
    'twenty-one': '21',
    'twenty-two': '22',
    'twenty-three': '23',
    'twenty-four': '24',
    'twenty-five': '25',
    'twenty-six': '26',
    'twenty-seven': '27',
    'twenty-eight': '28',
    'twenty-nine': '29',
    'thirty': '30',
    'yes': '是', 
    'no': '否',
    'airport':'机场',
    'airfield':'机场',
    'parking lot':'停车场',
    'car park':'停车场',
    'port':'港口',
    'village':'村庄',
    'countryside':'村庄',
    'city':'城市',
    'farmland':'农田',
    'airport terminal':'机场'
    } 

def translate_en_to_zh(model,tokenizer,article,task_name):

    # 检测是否为中文或者是阿拉伯数字，如果是，则直接返回
    if all(u'\u4e00' <= c <= u'\u9fff' or c.isdigit() for c in article):
        return article

    # 首先使用map匹配直接进行翻译，主要针对短文本进行翻译
    if task_name == "QA":
        article = article.strip().lower()
        article = article.strip('.')
        if article in translate_map:
            return translate_map[article]
    
    
    # 使用分词器对文本进行编码，将文本转换为模型输入所需的张量格式
    inputs = tokenizer(article, return_tensors="pt")
    
    # 生成翻译的令牌序列，强制生成中文简体(zho_Hans) 作为目标语言
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["zho_Hans"], max_length=30
    )
    
    # 将生成的令牌序列解码为可读的文本
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    
    # 打印翻译结果
    # print(translated_text)
    return translated_text