FROM tensorflow/tensorflow:1.7.0-devel-gpu-py3
LABEL maintainer "david lexuszhi1990@gmail.com"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential libatlas-base-dev libopencv-dev liblapack-dev \
        python-opencv libcurl4-openssl-dev libgtest-dev \
        cmake wget unzip curl git graphviz

RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple/ \
        keras opencv-python easydict cython configobj h5py \
        numpy nose-timer requests Pillow graphviz


