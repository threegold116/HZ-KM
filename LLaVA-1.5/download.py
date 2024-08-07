from huggingface_hub import snapshot_download
# import os
# os.environ['CURL_CA_BUNDLE'] = ''
from huggingface_hub import login
snapshot_download(repo_id='liuhaotian/llava-v1.5-7b',
                  repo_type='model',
                  local_dir='/home/wangzexin/model/llava-all-file',
                  force_download=True, resume_download=True,max_workers=1)
# export HF_ENDPOINT=https://hf-mirror.com
# pandagpt_13b_max_len_256
# pandagpt_13b_max_len_400

# 运行的命令是： HF_ENDPOINT=https://hf-mirror.com python download.py 设置环境变量
