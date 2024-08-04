from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
import torch
import sys

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
import os
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
    pattern=r"(.+)\'(.+)\'"
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 检查行是否非空
            if "image" in line: 
                key="image_path"
                # 分割键和值，假设键和值之间使用冒号和空格分隔
            elif "text" in line:
                key="text_input"
            else:
                continue
            # elif "question"in line:
            #     key="question_id"
            search_groups = re.search(pattern,line.strip())    
            if search_groups==None:
                continue
            if key not in result.keys():
                result[key] = []
            value = search_groups.group(2)
            value = value.strip()
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
def process_input(model_path,input_file_path,output_path):
    model_name=get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        attn_implementation=None
    )
    model.eval()

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
            # 图像读取
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
                image_path = image_path1 + ',' + image_path2
                promots.append(read_question_dict[prompt_key][0]+ "\n请用中文作答。")

            print(read_question_dict)
            try:
                for imput_prompt in promots:
                    print(imput_prompt)
                    args = type('Args', (), {
                    "model_path": model_path,
                    "model_base": None,
                    "model_name": get_model_name_from_path(model_path),
                    "query": imput_prompt,
                    "conv_mode": None,
                    "image_file": image_path,
                    "sep": ",",
                    "temperature": 0.8,
                    "top_p": None,
                    "num_beams": 1,
                    "max_new_tokens": 512
                    })()

                    outputs = eval_model(args,tokenizer, model, image_processor,model_name)
                    outputs = outputs.replace("答案：", "").replace("答案:", "")
                    
                    # 生成总结果
                    read_question_dict[answer_key].append(outputs)
                    print(read_question_dict)

                # 对于 Image_caption 和 Change_caption
                for id in read_question_dict[question_id_key]:
                    num_id = int(id)
                    if task_name == "QA":
                        output_res = output_res + "question_id:" + "'"+ id +"'" + "\n"                
                    output_res = output_res + "text_input:" + "'"+ read_question_dict[prompt_key][num_id] +"'" + "\n"
                    output_res = output_res + "text_truth:" + "'"+ read_question_dict[answer_key][num_id] +"'" + "\n"
            except Exception as e:
                for id in read_question_dict[question_id_key]:
                    num_id = int(id)
                    if task_name == "QA":
                        output_res = output_res + "question_id:" + "'"+ id +"'" + "\n"                
                    output_res = output_res + "text_input:" + "'"+ read_question_dict[prompt_key][0] +"'" + "\n"
                    output_res = output_res + "text_truth:" + "'是'" + "\n"

            output_res_path = os.path.join(output_path,task_name,file_name)
            os.makedirs(os.path.dirname(output_res_path),exist_ok=True)
            with open(output_res_path, 'w',encoding="utf-8") as file:
                file.write(output_res)
                

def eval_model(args,tokenizer, model, image_processor,model_name):
    # Model
    disable_torch_init()
    #FIXME:可以不用每次都加载模型
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    elif "next" in model_name.lower():
        conv_mode = "llava_llama_3"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode
    
    conv = conv_templates[args.conv_mode].copy()
    if conv_mode=="llava_llama_3":
        conv.tokenizer=tokenizer

    
    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = "\n".join([DEFAULT_IMAGE_TOKEN for size in image_sizes]+[qs])
    
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    print("*"*100)
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    print("*"*100)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=512,
        )
    print("*"*100)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)
    print("*"*100)
    return outputs


if __name__ == "__main__":

    # 检查命令行参数数量
    # if len(sys.argv) != 3:
    #     print("Usage: python run.py <input_path> <output_path>")
    #     sys.exit(1)

    # 命令行参数从 sys.argv[1] 开始，因为 sys.argv[0] 是脚本名称
    # test_data_path = sys.argv[1]
    # output_path = sys.argv[2]
    test_data_path = "/home/sxjiang/myproject/HZ/examples/input_path"
    output_path = "/home/sxjiang/myproject/HZ/examples/out_path"

    # llava-1.5预训练权重
    model_path = "/home/sxjiang/model/llama3-llava-next-8b"
    # test_data_path = "input_path"
    # output_path = "output_path"

    # 创建输出文件夹

    process_input(model_path,test_data_path,output_path)
   