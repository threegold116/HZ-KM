import pandas as pd
import os
import argparse
import shutil
import pandas as pd
import json
from baidu import baidu_api

import time





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="show information")
    parser.add_argument("--in_path","-i",type=str,default="/home/sxjiang/myproject/HZ/data/jsonfiles/vrs_eval_qa.json")
    parser.add_argument("--out_path","-ou",type=str,default="/home/sxjiang/myproject/HZ/data/jsonfiles/vrs_eval_qa_ch.json")
    args = parser.parse_args()
    if not os.path.exists(args.out_path):
        os.makedirs(os.path.dirname(args.out_path),exist_ok=True)
        f=open(args.out_path,"w")
        json.dump([],f)
        f.close()
        ch_data=[]
    else:
        with open(args.out_path,"r",encoding="utf-8") as f:
            ch_data=json.load(f)
    with open(args.in_path,"r",encoding="utf-8") as f:
        en_data=json.load(f)
    
    for index in range(len(ch_data),len(en_data)):
        line=en_data[index]
        new_line=line
        new_line["question"]=baidu_api(line["question"],from_lang='en',to_lang='zh')
        new_line["ground_truth"]=baidu_api(line["ground_truth"],from_lang='en',to_lang='zh')
        ch_data.append(new_line)
        f=open(args.out_path,"w")
        json.dump(ch_data,f,ensure_ascii=False)
        f.close()