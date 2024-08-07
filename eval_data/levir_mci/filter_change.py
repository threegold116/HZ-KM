import json
import os
#### VRS_Bench vqa.json过滤
ann_file_path="/home/jiangshixin/dataset/remote_sense/LEVIR-MCI/LEVIR-MCI-dataset/LevirCCcaptions.json"
with open(ann_file_path,"r") as f:
    data=json.load(f)
    data=data["images"]
# types={'image', 'object color', 'object size', 'object existence', 'object shape', 'rural or urban', 'object category', 'object quantity', 'scene type', 'object position', 'reasoning', 'object direction'}
# qa_types={"object color","object size","object existence","object shape","object category",'object quantity','scene type','object position','object direction','reasoning'}
# "several",
numbers = [
    "a ",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "twenty-one", "twenty-two", "twenty-three", "twenty-four", "twenty-five", "twenty-six", "twenty-seven", "twenty-eight", "twenty-nine", "thirty",
    "thirty-one", "thirty-two", "thirty-three", "thirty-four", "thirty-five", "thirty-six", "thirty-seven", "thirty-eight", "thirty-nine", "forty",
    "forty-one", "forty-two", "forty-three", "forty-four", "forty-five", "forty-six", "forty-seven", "forty-eight", "forty-nine", "fifty",
    "fifty-one", "fifty-two", "fifty-three", "fifty-four", "fifty-five", "fifty-six", "fifty-seven", "fifty-eight", "fifty-nine", "sixty",
    "sixty-one", "sixty-two", "sixty-three", "sixty-four", "sixty-five", "sixty-six", "sixty-seven", "sixty-eight", "sixty-nine", "seventy",
    "seventy-one", "seventy-two", "seventy-three", "seventy-four", "seventy-five", "seventy-six", "seventy-seven", "seventy-eight", "seventy-nine", "eighty",
    "eighty-one", "eighty-two", "eighty-three", "eighty-four", "eighty-five", "eighty-six", "eighty-seven", "eighty-eight", "eighty-nine", "ninety",
    "ninety-one", "ninety-two", "ninety-three", "ninety-four", "ninety-five", "ninety-six", "ninety-seven", "ninety-eight", "ninety-nine", "one hundred"
]

filter_data=[]
for line in data:
    # if line["type"] in qa_types:
    new_line=line
    new_line={}
    if line["filepath"]=="train":
        continue
    if line["changeflag"]==0:
        continue
    sentences=[]
    for sentence in line["sentences"]:
        for word in numbers:
            if word in sentence["raw"].lower():        
                sentences.append(sentence["raw"])
    biased_words=["some","several","many","rows","a row of","few","more","part","the row of"]
    new_sentences=[]
    for sentence in sentences:
        flag=False
        for word in biased_words:    
            if word in sentence.lower():
                flag=True
        if not flag:
            new_sentences.append(sentence)
    print(new_sentences)
    if len(new_sentences)==0:
        continue
    new_line["image_id"]=line["filename"]
    new_line["image_id"]=os.path.join(line["filepath"],line["filename"])
    new_line["question"]="NULL"
    lens=[len(sentence) for sentence in new_sentences]
    new_line["ground_truth"]=new_sentences[lens.index(max(lens))].strip()
    filter_data.append(new_line)
    
with open("/home/jiangshixin/myproject/HZ-KM/data/jsonfiles/levir_eval_change.json","w") as f:
    json.dump(filter_data,f)

