from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.mm_utils import get_model_name_from_path
from LLaVA.llava.eval.run_llava import eval_model
import torch
import sys

from LLaVA.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import (
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

def create_output_floder(output_path):
    # 定义需要创建的子目录列表
    sub_dirs = [
        'QA',
        'Image_caption',
        'Change_caption'
    ]

    # 创建根目录
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 在根目录下创建子目录
    for dir_name in sub_dirs:
        dir_path = os.path.join(output_path, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


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
def read_txt_as_dict(txt_file_path,task_name):
    result = {}
    delimiters = [':', "'","，","\"","“"]

    with open(txt_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 检查行是否非空
            if line:
                key = ""
                value = ""
                # 分割键和值，假设键和值之间使用冒号和空格分隔
                for delimiter in delimiters: 
                    try :
                        key, value = line.split(delimiter, 1)
                        break
                    except ValueError:
                        print("."*100)
                        print(line)
                        print("."*100)
                        continue
                if key == "" or value == "":
                    continue

                # 规范 key
                if (task_name == "QA" or task_name == "Image_caption") and ("image" in key):
                    key = "image_path"
                if "text_input" in key:
                    key = "text_input"
                if "text_truth" in key:
                    key = "text_truth"

                key = key.strip(":")
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
def process_imput(model_path,imput_file_path,output_path):

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )


    prompt_base = "你是一个遥感图像领域的专家，你能够针对高精度可见光遥感图像进行细致的分析，并且能够给出十分有用的答案。"

    for root, dirs, files in os.walk(imput_file_path):
        sub_floder = os.path.basename(root)
        if sub_floder in ['image1','image2','image']:
            continue
        # 只遍历 txt 文件
        for name in files:
            try:
                # 构建文件的完整路径
                file_path = os.path.join(root, name)
                file_path_parts = file_path.split('/')
                # 获得祖父文件夹名称，Change_caption、Image_caption 或者 QA
                task_name = file_path_parts[-3]
                file_name = file_path_parts[-1]
                read_question_dict = read_txt_as_dict(file_path,task_name)
                read_question_dict["text_truth"] = []
                output_res_path = os.path.join(output_path,task_name,file_name)
                # if os.path.exists(output_res_path):
                #     continue
                image_path = ""
                promots= []
                output_res = ""
                # 根据不同的任务名称进行txt文件读取
                print("-"*100)
                print(task_name)
                print(file_name)
                print(read_question_dict)
                print("-"*100)
                if task_name == "QA":
                    image_path = read_question_dict['image_path'][0]
                    output_res = "image_path:" + "'"+ image_path +"'"+ "\n"
                    for prompt_query in read_question_dict['text_input']:
                        promots.append(prompt_base + "\n 请用中文简要作答，直接给出答案，判断问题则直接回答“是”或者“否”，提问数量问题则直接回答数字。\n"  + prompt_query )

                if task_name == "Image_caption":
                    image_path = read_question_dict['image_path'][0]
                    output_res = "image_path:" + "'"+ image_path +"'"+ "\n"
                    promots.append(prompt_base + read_question_dict['text_input'][0] + "\n请用中文作答。")


                if task_name == "Change_caption":
                    image_path1 = read_question_dict['image_path1'][0]
                    output_res = "image_path1:" + "'"+ image_path1 +"'" + "\n"
                    image_path2 = read_question_dict['image_path2'][0]
                    output_res = output_res + "image_path2:" + "'"+ image_path2 +"'" + "\n"
                    image_path = image_path1 + ',' + image_path2
                    promots.append(prompt_base  + read_question_dict['text_input'][0]+ "\n请用中文作答。")
            except Exception as e:
                print(f"发生异常：{e}，但循环继续执行")
                with open(output_res_path, 'w',encoding="utf-8") as file:
                    file.write("")
                continue

            
            for imput_prompt in promots:
                args = type('Args', (), {
                "model_path": model_path,
                "model_base": None,
                "model_name": get_model_name_from_path(model_path),
                "query": imput_prompt,
                "conv_mode": None,
                "image_file": image_path,
                "sep": ",",
                "temperature": 0,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 512
                })()

                outputs = eval_model(args, tokenizer , model, image_processor , model_name)
                outputs = outputs.replace("答案：", "").replace("答案:", "")
                
                # 生成总结果
                read_question_dict["text_truth"].append(outputs)

            # 对于 Image_caption 和 Change_caption
            if len(promots) <= 1:
                output_res = output_res + "text_input:" + "'"+ read_question_dict['text_input'][0] +"'" + "\n"
                output_res = output_res + "text_truth:" + "'"+ read_question_dict["text_truth"][0] +"'" 
            else:
                for num_id in range(len(read_question_dict['text_input'])):
                    # num_id = int(id)
                    output_res = output_res + "question_id:" + "'"+ str(num_id) +"'" + "\n"
                    output_res = output_res + "text_input:" + "'"+ read_question_dict['text_input'][num_id] +"'" + "\n"
                    output_res = output_res + "text_truth:" + "'"+ read_question_dict["text_truth"][num_id] +"'" + "\n"

            
            print(output_res)
            print(output_res_path)
            with open(output_res_path, 'w',encoding="utf-8") as file:
                file.write(output_res)

def eval_model(args,tokenizer, model, image_processor,model_name):
    # Model
    disable_torch_init()

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
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    
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
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
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
    model_path = "llava-v1.5-7b"
    # test_data_path = "input_path"
    # output_path = "output_path"

    # 创建输出文件夹
    create_output_floder(output_path)

    process_imput(model_path,test_data_path,output_path)
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