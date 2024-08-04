import json
#### VRS_Bench vqa.json过滤
ann_file_path="/home/sxjiang/dataset/remote_sense/VRSBench/VRSBench_EVAL_vqa.json"
with open(ann_file_path,"r") as f:
    data=json.load(f)
types={'image', 'object color', 'object size', 'object existence', 'object shape', 'rural or urban', 'object category', 'object quantity', 'scene type', 'object position', 'reasoning', 'object direction'}
qa_types={"object color","object size","object existence","object shape","object category",'object quantity','scene type','object position','object direction','reasoning'}
filter_data=[]
for line in data:
    if line["type"] in qa_types:
        filter_data.append(line)
with open("/home/sxjiang/myproject/HZ/data/jsonfiles/vrs_qa.json","w") as f:
    json.dump(filter_data,f)

