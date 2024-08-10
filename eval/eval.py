from cider_eval import cider
from acc_eval import acc,llm_judge
import os
import re
import pandas as pd
import json
from tqdm import tqdm
# 阅读输入的 input的 txt文件
def read_answers_from_file(txt_file_path):
    answers = []
    pattern=r"(.+)\'(.+)\'"
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if "truth" not in line:
                continue 
            search_groups = re.search(pattern,line.strip())    
            if search_groups==None:
                answers.append("NULL")
                continue
            answer = search_groups.group(2)
            answer = answer.strip()
            answers.append(answer)
    return answers
# 阅读输入的 input的 txt文件
def read_questions_from_file(txt_file_path):
    questions = []
    pattern=r"(.+)\'(.+)\'"
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if "input" not in line:
                continue 
            search_groups = re.search(pattern,line.strip())    
            if search_groups==None:
                questions.append("NULL")
                continue
            answer = search_groups.group(2)
            answer = answer.strip()
            questions.append(answer)
    return questions



def eval(predict_path,gt_path,eval_result_path,judege_function):
    assert os.path.isdir(predict_path)
    task_names=["Change_caption","Image_caption","QA"]
    eval_result=[]
    #score=(原始cider+acc(%)+原始cider)/3
    bar=tqdm(total=3000)
    for root, dirs, files in os.walk(predict_path):
        if len(files)==0:
            continue
        for file in files:
            if int(file.split(".")[0])>=1000:
                continue
            bar.update(1)
            predict_file_path=os.path.join(root,file)
            task_name = predict_file_path.split("/")[-2]
            gt_file_path=os.path.join(gt_path,task_name,file)
            pred_answers=read_answers_from_file(predict_file_path)
            questions=read_questions_from_file(gt_file_path)
            gt_answers=read_answers_from_file(gt_file_path)
            if len(pred_answers)!=len(gt_answers):#FIXME:当数量不相同时的处理方法
                eval_result.append({"id":task_name+" "+file.split(".")[0],"pred":"","gt":"","correct":0,"sum":len(gt_answers),"cider":0,"task":task_name})            
            file_score={"id":task_name+" "+file.split(".")[0],"pred":pred_answers,"gt":gt_answers,"correct":0,"question_nums":len(gt_answers),"cider":0,"task":task_name}
            for index in range(len(gt_answers)):
                gt_answer=gt_answers[index]
                pred_answer=pred_answers[index]
                question=questions[index]
                if task_name=="QA":
                    if judege_function=="llm_judge":
                        file_score["correct"]+=llm_judge(pred_answer,gt_answer,question)
                    else:
                        file_score["correct"]+=acc(pred_answer,gt_answer)
                else:
                    file_score["cider"]+=cider(pred_answer,gt_answer)
            eval_result.append(file_score)
    #写入json文件
    os.makedirs(eval_result_path,exist_ok=True)
    with open(os.path.join(eval_result_path,"judge.json"),"w",encoding="utf-8") as f:
        json.dump(eval_result,f)
    #用pandas读入，便于分组处理
    eval_pd_result = pd.read_json(os.path.join(eval_result_path,"judge.json"))
    eval_pd_result.to_csv(os.path.join(eval_result_path,"judge.csv"))
    eval_pd_result_task_groups=eval_pd_result.groupby("task")
    task_scores=[]
    for task,group in eval_pd_result_task_groups:
        num_sum=0
        score_sum=0
        for _,line in group.iterrows():
            num_sum+=line["question_nums"]
            if task=="QA":
                score_sum+=(line["correct"]*100)#因为acc显示的是%
            else:
                score_sum+=line["cider"]
        task_score={"type":task,"score":score_sum/num_sum}
        task_scores.append(task_score)
    av_score=0
    for task_score in task_scores:
        av_score+=task_score["score"]
    av_score=av_score/len(task_names)
    task_scores.append({"type":"avg","score":av_score})
    #写入json文件
    print(eval_result_path)
    with open(os.path.join(eval_result_path,"score.json"),"w") as f:
        json.dump(task_scores,f)
    bar.close()
    
                    
            
        





if __name__=="__main__":
    # 检查命令行参数数量
    # if len(sys.argv) != 3:
    #     print("Usage: python run.py <input_path> <output_path>")
    #     sys.exit(1)

    # 命令行参数从 sys.argv[1] 开始，因为 sys.argv[0] 是脚本名称
    # test_data_path = sys.argv[1]
    # output_path = sys.argv[2]

    gt_path = "/home/jiangshixin/dataset/remote_sense/VAL_HZPC/gt"
    predict_path = "/home/jiangshixin/dataset/remote_sense/VAL_HZPC/LLaVA-Next/origin"
    eval_result_path="/home/jiangshixin/myproject/HZ-KM/result/LLaVA-Next/debug"
    judege_function = "llm_judge"
    # judege_function = "rule_judge"
    eval(predict_path=predict_path,gt_path=gt_path,eval_result_path=eval_result_path,judege_function=judege_function)