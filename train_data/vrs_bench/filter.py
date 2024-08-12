import json
import re
import os
#### VRS_Bench vqa.json过滤
ann_file_path="/home/jiangshixin/dataset/remote_sense/VRSBench/VRSBench_train.json"
image_dir="/home/jiangshixin/dataset/remote_sense/VRSBench/Images_train"
with open(ann_file_path,"r") as f:
    data=json.load(f)
# types={'image', 'object color', 'object size', 'object existence', 'object shape', 'rural or urban', 'object category', 'object quantity', 'scene type', 'object position', 'reasoning', 'object direction'}
# qa_types={"object color","object size","object existence","object shape","object category",'object quantity','scene type','object position','object direction','reasoning'}
# train_types={'0.51, 0.2, 0.55, 0.3', 'refer', 'caption', 'vqa'}
train_types={'caption', 'vqa'}
filter_data=[]
count=0
for line in data:
    pattern=r"(.+)\[(.+)\](.+)"
    new_line=line
    conversation=line["conversations"]
    conversation=json.dumps(conversation)
    if "refer" in conversation:
        continue
    if "[caption]" in conversation:
        conversation=conversation.replace("[caption]","")
        continue
    if "[vqa]" in conversation:
        conversation=conversation.replace("[vqa]","")
    conversation=conversation.replace('"from"','"role"')
    
    conversation=conversation.replace('"gpt"','"assistant"')
    conversation=conversation.replace('"human"','"user"')
    conversation=conversation.replace('"value"','"content"')
    new_line["conversations"]=eval(conversation)
    new_line["image"]=os.path.join(image_dir,line["image"])
    assert os.path.exists(new_line["image"])
    filter_data.append(new_line)
    count+=1

print(train_types)          
    # if line["type"] in qa_types:

result_dir="/home/jiangshixin/myproject/HZ-KM/train_data/jsonfiles"
os.makedirs(result_dir,exist_ok=True)

with open("/home/jiangshixin/myproject/HZ-KM/train_data/jsonfiles/vrs_train_qa.json","w") as f:
    print(len(filter_data))
    json.dump(filter_data,f)

