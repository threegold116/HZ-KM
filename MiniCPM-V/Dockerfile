# 基础镜像若使用 cuda，需满足 cuda 版本高于等于 10.0，低于 11.7
# 示例如下
FROM registry.cn-hangzhou.aliyuncs.com/tian_zhi_bei_wzx/cuda:12.1.1-cudnn8-devel-ubuntu20.04
# 设置使用国内源
RUN mkdir -p /etc/apt
RUN touch /etc/apt/sources.list
RUN sed -i "s/archive.ubuntu./mirrors.aliyun./g" /etc/apt/sources.list
RUN sed -i "s/deb.debian.org/mirrors.aliyun.com/g" /etc/apt/sources.list
RUN sed -i "s/security.debian.org\/debian-security/mirrors.aliyun.com\/debian-security/g" /etc/apt/sources.list
RUN sed -i "s/httpredir.debian.org\/debian-security/mirrors.aliyun.com\/debian-security/g" /etc/apt/sources.list

# Ubuntu系统上运行 apt-get 命令时禁用交互式提示和用户输入
ENV DEBIAN_FRONTEND=noninteractive
# 配置程序依赖环境
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        git \
        vim \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        python3.10 \
        python3-dev \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN rm -rf /usr/bin/python
RUN ln -s /usr/bin/python3 /usr/bin/python

# 设置pip国内源
RUN pip install -U pip
RUN pip config set global.index-url http://pypi.tuna.tsinghua.edu.cn/simple
RUN pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

# 将程序复制容器内，表示在/workspace 路径下，使得后续能够用 pip3 install -e .
COPY . /workspace


# pip安装包
RUN pip3 install --upgrade pip
RUN cd /workspace && pip3 install -r requirements.txt
RUN pip3 install opencv-python==4.8.0.74
RUN pip3 install scikit-image
RUN pip3 install mmsegmentation==0.13.0
RUN pip3 install pytest-runner
RUN pip3 install mmcv==1.3.1




# 确定容器启动时程序运行路径
WORKDIR /workspace
# 确定容器启动命令。以 python 示例，python 表示编译器，run.py 表示执行文件，/input_path 和/output_path 为容器内绝对路径，测评时会自动将测试数据挂载到容器内/input_path 路径，无需修改
CMD ["python", "run.py", "/input_path", "/output_path"]