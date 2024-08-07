import pandas as pd
import os
import argparse
import shutil
import pandas as pd
import json
import time
import shutil
import random
cap_questions=["请用一段话介绍这张图片的内容。","请简要描述一下图像的内容。","请概述这张图片的内容。","请简单地介绍一下这张图片。"]

change_questions=["这两张图片有哪些区别？","两张图像有何差异？","这两张图像之间有什么区别？","请说明这两张图片的不同之处。"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="show information")
    parser.add_argument("--in_path","-i",type=str,default="/home/jiangshixin/myproject/HZ-KM/eval_data/jsonfiles/vrs_eval_cap_ch.json")
    parser.add_argument("--out_path","-ou",type=str,default="/home/jiangshixin/dataset/remote_sense/VAL_HZPC")
    parser.add_argument("--task","-ta",type=str,default="Image_caption")
    parser.add_argument("--image_dir","-im",type=str,default="/home/jiangshixin/dataset/remote_sense/VRSBench/Images_val")
    args = parser.parse_args()
    os.makedirs(args.out_path,exist_ok=True)
    with open(args.in_path,"r",encoding="utf-8") as f:
        json_data=json.load(f)
    pd_data=pd.read_json(args.in_path)
    pd_group_data=pd_data.groupby("image_id")
    index=0    
    for image_id,group in pd_group_data:
        image=image_id
        if args.task!="Change_caption":
            image_source_path=os.path.join(args.image_dir,image)    
            image_dst_path=os.path.join(args.out_path,"input_path",args.task,"image",str(index)+os.path.splitext(image)[-1])
            os.makedirs(os.path.dirname(image_dst_path),exist_ok=True)
            shutil.copy(image_source_path,image_dst_path)
        else:
            image_source_path_1=os.path.join(args.image_dir,image.split("/")[0],"A",image.split("/")[1])    
            image_dst_path_1=os.path.join(args.out_path,"input_path",args.task,"image1",str(index)+os.path.splitext(image)[-1])
            os.makedirs(os.path.dirname(image_dst_path_1),exist_ok=True)
            shutil.copy(image_source_path_1,image_dst_path_1)
            image_source_path_2=os.path.join(args.image_dir,image.split("/")[0],"B",image.split("/")[1])    
            image_dst_path_2=os.path.join(args.out_path,"input_path",args.task,"image2",str(index)+os.path.splitext(image)[-1])
            os.makedirs(os.path.dirname(image_dst_path_2),exist_ok=True)
            shutil.copy(image_source_path_2,image_dst_path_2)
        questions=[]
        answers=[]
        for _,line in group.iterrows():
            if args.task=="Image_caption":
                question=random.sample(cap_questions,1)[0]
            if args.task=="Change_caption":
                question=random.sample(change_questions,1)[0]   
            if  args.task=="QA":
                question=line["question"]
            if question in questions:
                continue
            questions.append(question)
            answer=line["ground_truth"]
            answers.append(answer)

        question_dst_path=os.path.join(args.out_path,"input_path",args.task,"question",str(index)+".txt")
        os.makedirs(os.path.dirname(question_dst_path),exist_ok=True)
        with open(question_dst_path,"w",encoding="utf-8") as f:
            if args.task!="Change_caption":            
                f.write("image_path:"+"'....../"+str(index)+os.path.splitext(image)[-1]+"'\n")
            else:
                f.write("image_path1:"+"'....../"+str(index)+os.path.splitext(image)[-1]+"'\n")
                f.write("image_path2:"+"'....../"+str(index)+os.path.splitext(image)[-1]+"'\n")
            for question_id in range(len(questions)):
                if args.task=="QA":
                    f.write("question_id:"+str(question_id)+"\n")
                f.write("text_input:"+"'"+questions[question_id]+"'\n")
                f.write("text_truth:\n")
        gt_dst_path=os.path.join(args.out_path,"gt",args.task,str(index)+".txt")
        os.makedirs(os.path.dirname(gt_dst_path),exist_ok=True)
        with open(gt_dst_path,"w",encoding="utf-8") as f:
            if args.task!="Change_caption":            
                f.write("image_path:"+"'....../"+str(index)+os.path.splitext(image)[-1]+"'\n")
            else:
                f.write("image_path1:"+"'....../"+str(index)+os.path.splitext(image)[-1]+"'\n")
                f.write("image_path2:"+"'....../"+str(index)+os.path.splitext(image)[-1]+"'\n")
            for question_id in range(len(questions)):
                if args.task=="QA":
                    f.write("question_id:"+str(question_id)+"\n")
                f.write("text_input:"+"'"+questions[question_id]+"'\n")
                f.write("text_truth:"+"'"+answers[question_id]+"'\n")
                       
        
        index+=1
        if index==1000:
            break
    # for index in range(len(json_data)):
    #     line=json_data[index]
    #     question=line["question"]
    #     answer=line["ground_truth"]
    #     image=line["image_id"]
    #     image_source_path=os.path.join(args.image_dir,image)    
    #     image_dst_path=os.path.join(args.out_path,"input_path",args.task,"image",str(index)+os.path.splitext(image)[-1])
    #     os.makedirs(os.path.dir(image_dst_path),exist_ok=True)
    #     shutil.copy(image_source_path,image_dst_path)
    #     question_dst_path=os.path.join(args.out_path,"input_path",args.task,"image",str(index)+os.path.splitext(image)[-1])
        