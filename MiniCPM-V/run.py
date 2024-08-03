
from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
import os
import torch
from transformers import AutoModel, AutoTokenizer
from MiniCPM import MiniCPMV,MiniCPMVTokenizerFast,MiniCPMVProcessor,MiniCPMVImageProcessor
import sys

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

    with open(txt_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 检查行是否非空
            if line:
                # 分割键和值，假设键和值之间使用冒号和空格分隔
                key, value = line.split(':', 1)
                if key not in result.keys():
                    result[key] = []
                value = value.strip()
                # 去除值两侧的单引号
                value = value.strip("'")
                # 将键值对添加到字典中
                result[key].append(value)
    txt_file_grandfather_dir=os.path.dirname(os.path.dirname(txt_file_path))
    for key,value in result.items():
        if "image" in key:
            result[key][0]=os.path.join(txt_file_grandfather_dir,key.replace("_path",""),value[0].split("/")[-1])
        
    return result

# 给定输入进行分类处理
def test(model_path,input_file_path,output_path):
    model = MiniCPMV.from_pretrained(model_path,  torch_dtype=torch.float16)
    model = model.to(device='cuda')

    tokenizer = MiniCPMVTokenizerFast.from_pretrained(model_path)
    model.eval()
    processor = MiniCPMVProcessor(image_processor=MiniCPMVImageProcessor.from_pretrained(model_path),
                              tokenizer=MiniCPMVTokenizerFast.from_pretrained(model_path))

    prompt_base = "你是一个遥感图像领域的专家，你能够针对高精度可见光遥感图像进行细致的分析，并且能够给出十分有用的答案。"

    for root, dirs, files in os.walk(input_file_path):
        sub_floder = os.path.basename(root)
        if sub_floder in ['image1','image2','image']:
            continue
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
            output_res = ""
            # 根据不同的任务名称进行txt文件读取
            image_keys=[]
            for key,_ in  read_question_dict.items():
                if "image" in key:
                    image_keys.append(key)
                    
            if task_name == "QA":
                image_path = read_question_dict[image_keys[0]][0]
                output_res = "image_path:" + "'"+ image_path +"'"+ "\n"
                for prompt_query in read_question_dict['text_input']:
                    promots.append(prompt_base + "\n 请用中文简要作答，直接给出答案，判断问题则直接回答“是”或者“否”，提问数量问题则直接回答数字。"  + prompt_query )
            if task_name == "Image_caption":
                image_path = read_question_dict[image_keys[0]][0]
                output_res = "image_path:" + "'"+ image_path +"'"+ "\n"
                promots.append(prompt_base + read_question_dict['text_input'][0] + "\n请用中文作答。")


            if task_name == "Change_caption":
                image_path1 = read_question_dict[image_keys[0]][0]
                output_res = "image_path1:" + "'"+ image_path1 +"'" + "\n"
                image_path2 = read_question_dict[image_keys[1]][0]
                output_res = output_res + "image_path2:" + "'"+ image_path2 +"'" + "\n"
                image_path = image_path1 + ',' + image_path2
                promots.append(prompt_base  + read_question_dict['text_input'][0]+ "\n请用中文作答。")

            print(read_question_dict)
            try:
                for imput_prompt in promots:
                    print(imput_prompt)
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
                    read_question_dict["text_truth"].append(outputs)
                    print(read_question_dict)

                # 对于 Image_caption 和 Change_caption
                if len(promots) <= 1:
                    output_res = output_res + "text_input:" + "'"+ read_question_dict['text_input'][0] +"'" + "\n"
                    output_res = output_res + "text_truth:" + "'"+ read_question_dict["text_truth"][0] +"'" 
                else:
                    for id in read_question_dict['question_id']:
                        num_id = int(id)
                        output_res = output_res + "question_id:" + "'"+ id +"'" + "\n"
                        output_res = output_res + "text_input:" + "'"+ read_question_dict['text_input'][num_id] +"'" + "\n"
                        output_res = output_res + "text_truth:" + "'"+ read_question_dict["text_truth"][num_id] +"'" + "\n"

            except Exception as e:
                if len(promots) <= 1:
                    output_res = output_res + "text_input:" + "'"+ read_question_dict['text_input'][0] +"'" + "\n"
                    output_res = output_res + "text_truth:" + "'Output'" 
                else:
                    for id in read_question_dict['question_id']:
                        num_id = int(id)
                        output_res = output_res + "question_id:" + "'"+ id +"'" + "\n"
                        output_res = output_res + "text_input:" + "'"+ read_question_dict['text_input'][num_id] +"'" + "\n"
                        output_res = output_res + "text_truth:" + "'是'" + "\n"

            output_res_path = os.path.join(output_path,task_name,file_name)
            os.makedirs(os.path.dirname(output_res_path),exist_ok=True)
            with open(output_res_path, 'w',encoding="utf-8") as file:
                file.write(output_res)

def eval_model(args,tokenizer, model, processor):
    # Model    
    image_files = image_parser(args)
    images = load_images(image_files)
    qs = args.query
    print("*"*100)
    msgs = [{'role': 'user', 'content': qs}]
    print("*"*100)
    outputs = model.chat(
        image=images,
        msgs=msgs,
        processor=processor,
        tokenizer=tokenizer,
        sampling=False, # if sampling=False, beam_search will be used by default
        temperature=0.7,
        system_prompt='你是一个遥感图像领域的专家，你能够针对高精度可见光遥感图像进行细致的分析，并且能够给出具体的答案。' # pass system_prompt if needed
    )
    print("*"*100)
    print(outputs)
    print("*"*100)
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
    model_path = "/home/sxjiang/model/MiniCPM-Llama3-V-2_5"
    # test_data_path = "input_path"
    # output_path = "output_path"


    test(model_path,test_data_path,output_path)
    # exit()


    # # 文本提示
    # prompt = "这两张图片有哪些区别？请用中文作答"
    # # 测试图片路径
    # image_file = "/home/zxwang/module/llava-all-file/input_path/Change_caption/image1/0.png,/home/zxwang/module/llava-all-file/input_path/Change_caption/image2/0.png"

    # args = type('Args', (), {
    #     "model_path": model_path,
    #     "model_base": None,
    #     "model_name": get_model_name_from_path(model_path),
    #     "query": prompt,
    #     "conv_mode": "vicuna_v1",
    #     "image_file": image_file,
    #     "sep": ",",
    #     "temperature": 0,
    #     "top_p": None,
    #     "num_beams": 1,
    #     "max_new_tokens": 512
    # })()

    # outputs = eval_model(args)
    # print(outputs)