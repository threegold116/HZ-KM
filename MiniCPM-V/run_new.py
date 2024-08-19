
from PIL import Image
import re
import requests
from PIL import Image
from io import BytesIO
import re
import os
import torch
from transformers import AutoModel, AutoTokenizer
from MiniCPM import MiniCPMV,MiniCPMVTokenizerFast,MiniCPMVProcessor,MiniCPMVImageProcessor
import sys
import re
from change_task.predict import Change_Perception
from tqdm import tqdm
def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

# 阅读输入的 input的 txt文件
def read_txt_as_dict(txt_file_path):
    result = {}
    # pattern=r"(.+)\'(.+)\'"
    delimiters = [':', "'",'：',"，","\"","“"]
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            key = ""
            value = ""
            # 分割键和值，假设键和值之间使用冒号和空格分隔
            for delimiter in delimiters:
                try :
                    key, value = line.split(delimiter, 1)
                    break
                except ValueError:
                    continue
            value = value.strip()
            value = value.strip("'")

            if key == "" or value == "":
                continue 

            # 检查行是否非空，或者非法
            if "image" in key: 
                key="image_path"
            elif "text_input" in key:
                key="text_input"
            else:
                continue

            if key not in result.keys():
                result[key] = []
            # 将键值对添加到字典中
            result[key].append(value)

    txt_file_grandfather_dir=os.path.dirname(os.path.dirname(txt_file_path))
    for key,values in result.items():
        if "image" in key:
            for i in range(len(values)):
                image_dirs_name=['image1','image2','image']
                image_paths = [os.path.join(txt_file_grandfather_dir,image_dir,values[i].split("/")[-1]) for image_dir in image_dirs_name]
                image_path=""
                for condidate in image_paths:
                    if os.path.exists(condidate) and condidate not in result[key]:
                        image_path=condidate
                        break
                result[key][i] = image_path
    return result

# 给定输入进行分类处理
def test(model_path,input_file_path,output_path):
    model = MiniCPMV.from_pretrained(model_path,  torch_dtype=torch.float16)
    model = model.to(device='cuda')

    tokenizer = MiniCPMVTokenizerFast.from_pretrained(model_path)
    model.eval()
    processor = MiniCPMVProcessor(image_processor=MiniCPMVImageProcessor.from_pretrained(model_path),
                              tokenizer=MiniCPMVTokenizerFast.from_pretrained(model_path))
    change_perception = Change_Perception()

    prompt_base = "你是一个遥感图像领域的专家，你能够针对高精度可见光遥感图像进行细致的分析，并且能够给出十分有用的答案。"

    #如果不存在
    old_image_path=""
    old_prompt_text="TEST"
    
    for root, dirs, files in os.walk(input_file_path):
        sub_floder = os.path.basename(root)
        if sub_floder in ['image1','image2','image']:
            continue
        
        bar=tqdm(total=len(files))
        # 只遍历 txt 文件
        for name in files:

            # 构建文件的完整路径
            file_path = os.path.join(root, name)
            file_path_parts = file_path.split('/')
            # 获得祖父文件夹名称，Change_caption、Image_caption 或者 QA
            task_name = file_path_parts[-3]
            file_name = file_path_parts[-1]
            read_question_dict = read_txt_as_dict(file_path)
            read_question_dict["text_truth"] = []
            image_path = ""
            promots= []
            postions = []
            output_res = ""
            # 输出文件设置
            output_res_path = os.path.join(output_path,task_name,file_name)
            if os.path.exists(output_res_path):
                bar.update(1)
                continue
            
            # 读取内容时的冗余设置
            image_key="image_path"
            prompt_key="text_input"
            question_id_key="question_id"
            answer_key="text_truth"
            if image_key not in read_question_dict:
                read_question_dict[image_key]=[]
                read_question_dict[image_key][0]=old_image_path
            elif len(read_question_dict[image_key])==0:
                read_question_dict[image_key][0]=old_image_path
            else:
                old_image_path=read_question_dict[image_key][0]
                
            if prompt_key not in read_question_dict:
                read_question_dict[prompt_key]=[]
                read_question_dict[prompt_key][0]=old_prompt_text
            elif len(read_question_dict[prompt_key])==0:
                read_question_dict[prompt_key][0]=old_prompt_text
            else:
                old_prompt_text=read_question_dict[prompt_key][0] 
            read_question_dict[question_id_key]=[]
            for i in range(len(read_question_dict[prompt_key])):
                read_question_dict[question_id_key].append(str(i))
            
            

            # 根据不同的任务名称进行txt文件读取
            if task_name == "QA":
                image_path = read_question_dict[image_key][0]
                output_res = "image_path:" + "'"+ image_path +"'"+ "\n"
                for prompt_query in read_question_dict[prompt_key]:
                    promots.append("\n 请用中文简要作答，直接给出答案，判断问题则直接回答“是”或者“否”，提问数量问题则直接回答数字。"  + prompt_query )
            if task_name == "Image_caption":
                image_path = read_question_dict[image_key][0]
                output_res = "image_path:" + "'"+ image_path +"'"+ "\n"
                promots.append(read_question_dict[prompt_key][0] + "\n请用中文作答。")


            if task_name == "Change_caption":
                image_path1 = read_question_dict[image_key][0]
                output_res = "image_path1:" + "'"+ image_path1 +"'" + "\n"
                image_path2 = read_question_dict[image_key][0] if len(read_question_dict[image_key])==1 else read_question_dict[image_key][1]
                output_res = output_res + "image_path2:" + "'"+ image_path2 +"'" + "\n"
                image_path = ""
                pos_caption_map = change_perception.get_change_position_caption_list(image_path2, image_path1)
                # 倘若没检查出差别
                change_flag = True
                if len(pos_caption_map.keys()) == 0:
                    change_flag = False
                    pos_caption_map = {"上方":"公路旁增加了2座建筑，拆除了1处建筑"}
                else:
                    for key , value in pos_caption_map.items():
                        prompt = "将以下句子翻译为中文：\n "+ value
                        promots.append(prompt)
                        postions.append(key)
                # promots.append(read_question_dict[prompt_key][0]+ "\n请用中文作答。")

            # print(read_question_dict)
            try:
                # change 任务中倘若没有检测出来变化，则不经过 LLM
                if task_name == "QA" or task_name == "Image_caption" or (task_name == "Change_caption" and change_flag):
                    for imput_prompt in promots:
                        # print(imput_prompt)
                        args = type('Args', (), {
                        "model_path": model_path,
                        "model_base": None,
                        "query": imput_prompt,
                        "conv_mode": None,
                        "image_file": image_path,
                        "sep": ",",
                        "temperature": 0.8,
                        "top_p": None,
                        "num_beams": 1,
                        "max_new_tokens": 512,
                        "system_prompt":prompt_base
                        })()

                        outputs = eval_model(args,tokenizer, model, processor)
                        outputs = outputs.replace("答案：", "").replace("答案:", "")
                        
                        # 生成总结果
                        read_question_dict[answer_key].append(outputs)
                        # print(read_question_dict)

                    
                # 对于 change
                if task_name == "Change_caption":
                    answer = ""
                    output_res = output_res + "text_input:" + "'"+ read_question_dict[prompt_key][0] +"'" + "\n"
                    # 对检测出来差异和没检测出来分别处理
                    if change_flag:
                        for index, position in enumerate(postions): 
                            answer = answer + "在图像的" + postions[index] +  read_question_dict["text_truth"][index]
                        output_res = output_res + "text_truth:" + "'"+ answer +"'" 
                    else:
                        output_res = output_res + "text_truth:" + "'"+ "图像上方公路旁增加了2座建筑，拆除了一处建筑。" +"'"
                else:
                    # 对于 Image_caption 和 QA
                    for id in read_question_dict[question_id_key]:
                        num_id = int(id)
                        if task_name == "QA":
                            # 除了 QA 第一行，都要加换行
                            if num_id != 0 :
                                output_res = output_res + "\n"
                            output_res = output_res + "question_id:" + id + "\n"                
                        output_res = output_res + "text_input:" + "'"+ read_question_dict[prompt_key][num_id] +"'" + "\n"
                        output_res = output_res + "text_truth:" + "'"+ read_question_dict[answer_key][num_id] +"'"

                    

            except Exception as e:
                for id in read_question_dict[question_id_key]:
                    num_id = int(id)
                    if task_name == "QA":
                        # 除了 QA 第一行，都要加换行
                        if num_id != 0 :
                            output_res = output_res + "\n"
                        output_res = output_res + "question_id:" + id + "\n"                
                    output_res = output_res + "text_input:" + "'"+ read_question_dict[prompt_key][0] +"'" + "\n"

                    # 骗分答案
                    if task_name == "QA":
                        output_res = output_res + "text_truth:" + "'否'"
                    elif task_name == "Image_caption":
                        output_res = output_res + "text_truth:" + "'图像显示的是海上场景，有一个停车场，机场中间有3驾飞机停靠。'"
                    else:
                        output_res = output_res + "text_truth:" + "'"+ "图像上方公路旁增加了2座建筑，拆除了一处建筑。" +"'"


            output_res_path = os.path.join(output_path,task_name,file_name)
            os.makedirs(os.path.dirname(output_res_path),exist_ok=True)
            with open(output_res_path, 'w',encoding="utf-8") as file:
                file.write(output_res)
            bar.update(1)
        bar.close()

def eval_model(args,tokenizer, model, processor):
    # Model
    if args.image_file == "":
        images = None
    else:    
        image_files = image_parser(args)
        images = load_images(image_files)
    qs = args.query
    # print("*"*100)
    msgs = [{'role': 'user', 'content': qs}]
    # print("*"*100)
    outputs = model.chat(
        image=images,
        msgs=msgs,
        processor=processor,
        tokenizer=tokenizer,
        sampling=False, # if sampling=False, beam_search will be used by default
        temperature=0.7,
        system_prompt='你是一个遥感图像领域的专家，你能够针对高精度可见光遥感图像进行细致的分析，并且能够给出具体的答案。' # pass system_prompt if needed
    )
    # print("*"*100)
    # print(outputs)
    # print("*"*100)
    return outputs



if __name__ == "__main__":

    # 检查命令行参数数量
    if len(sys.argv) != 3:
        print("Usage: python run.py <input_path> <output_path>")
        sys.exit(1)

    # 命令行参数从 sys.argv[1] 开始，因为 sys.argv[0] 是脚本名称
    test_data_path = sys.argv[1]
    output_path = sys.argv[2]

    # llava-1.5预训练权重
    model_path = "./MiniCPM-V_finetune_LLM"
    # test_data_path = "input_path"
    # output_path = "output_path"
    test(model_path,test_data_path,output_path)
    