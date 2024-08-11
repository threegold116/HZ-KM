# test.py
import torch
from transformers import AutoProcessor
from PIL import Image
from MiniCPM import MiniCPMV,MiniCPMVTokenizerFast,MiniCPMVProcessor,MiniCPMVImageProcessor
model_path="/home/jiangshixin/model/minicpmv/test_hz/minicpmv2_merge"
model = MiniCPMV.from_pretrained(model_path,  torch_dtype=torch.float16)
model = model.to(device='cuda')

tokenizer = MiniCPMVTokenizerFast.from_pretrained(model_path)
model.eval()
image1 = Image.open('/home/jiangshixin/myproject/HZ-KM/examples/input_path/Image_caption/image/0.tif').convert('RGB')
# image2 = Image.open('/home/sxjiang/project/LLaVA-NeXT/test/input_path/Change_caption/image2/0.png').convert('RGB')
question = '请问？'
image = [image1]
processor = MiniCPMVProcessor(image_processor=MiniCPMVImageProcessor.from_pretrained(model_path),
                              tokenizer=MiniCPMVTokenizerFast.from_pretrained(model_path))
question="你好！"
while question!="quit":
    msgs = [{'role': 'user', 'content': question}]
    res = model.chat(
        image=image,
        msgs=msgs,
        processor=processor,
        tokenizer=tokenizer,
        sampling=False, # if sampling=False, beam_search will be used by default
        temperature=0.7,
        system_prompt='' # pass system_prompt if needed
    )
    print(res)
    question=input("用户输入:")
    if question=="image":
        image_path=input("图像路径:")    
        image=[Image.open(image_path).convert('RGB')]
        question=input("用户输入:")
        

