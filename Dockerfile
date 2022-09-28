FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
RUN apt-get -y install python3.9 python3.9-distutils python3-pip
RUN python3.9 -m pip install -U pip wheel setuptools
RUN python3.9 -m pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN python3.9 -m pip install pandas matplotlib black ipykernel