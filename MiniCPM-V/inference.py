# test.py
import torch
from transformers import AutoProcessor
from PIL import Image
from MiniCPM import MiniCPMV,MiniCPMVTokenizerFast,MiniCPMVProcessor,MiniCPMVImageProcessor
model = MiniCPMV.from_pretrained('/home/sxjiang/model/MiniCPM-Llama3-V-2_5',  torch_dtype=torch.float16)
model = model.to(device='cuda')

tokenizer = MiniCPMVTokenizerFast.from_pretrained('/home/sxjiang/model/MiniCPM-Llama3-V-2_5')
model.eval()
image1 = Image.open('/home/sxjiang/project/LLaVA-NeXT/test/input_path/Change_caption/image1/0.png').convert('RGB')
# image2 = Image.open('/home/sxjiang/project/LLaVA-NeXT/test/input_path/Change_caption/image2/0.png').convert('RGB')
question = '请问？'
image = [image1]
msgs = [{'role': 'user', 'content': question}]
processor = MiniCPMVProcessor(image_processor=MiniCPMVImageProcessor.from_pretrained('/home/sxjiang/model/MiniCPM-Llama3-V-2_5'),
                              tokenizer=MiniCPMVTokenizerFast.from_pretrained('/home/sxjiang/model/MiniCPM-Llama3-V-2_5'))

res = model.chat(
    image=image,
    msgs=msgs,
    processor=processor,
    tokenizer=tokenizer,
    sampling=False, # if sampling=False, beam_search will be used by default
    temperature=0.7,
    system_prompt='你是一个遥感图像领域的专家，你能够针对高精度可见光遥感图像进行细致的分析，并且能够给出十分有用的答案。' # pass system_prompt if needed
)
print(res)

