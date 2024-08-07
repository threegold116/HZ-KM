import json
#### VRS_Bench vqa.json过滤
ann_file_path="/home/jiangshixin/dataset/remote_sense/VRSBench/VRSBench_EVAL_Cap.json"
with open(ann_file_path,"r") as f:
    data=json.load(f)
# types={'image', 'object color', 'object size', 'object existence', 'object shape', 'rural or urban', 'object category', 'object quantity', 'scene type', 'object position', 'reasoning', 'object direction'}
# qa_types={"object color","object size","object existence","object shape","object category",'object quantity','scene type','object position','object direction','reasoning'}
filter_data=[]
for line in data:
    # if line["type"] in qa_types:
    new_line=line
    biased_words=["from GoogleEarth","high-resolution","Google Earth","Aerial","aerial","via GoogleEarth","by GoogleEarth","The source of the image is GoogleEarth","the source of the image is GoogleEarth","GoogleEarth","sourced with high resolution","features high resolution","with high resolution","in high resolution","high resolution"]
    for word in biased_words:
        new_line["ground_truth"]=new_line["ground_truth"].replace(word,"")
    new_line["ground_truth"]=new_line["ground_truth"].replace("  "," ")
    filter_data.append(new_line)
    
with open("/home/jiangshixin/myproject/HZ-KM/data/jsonfiles/vrs_cap.json","w") as f:
    json.dump(filter_data,f)

