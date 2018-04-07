FROM nvidia/cuda:9.0-cudnn7-runtime
LABEL maintainer "david lexuszhi1990@gmail.com"

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:jonathonf/python-3.6 && \
    apt-get update && \
    apt-get install -y python3.6-dev libgfortran3 python3-setuptools

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6 && \
    echo "PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/dist-packagesls" > /root/.bashrc

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential libatlas-base-dev libopencv-dev liblapack-dev \
        python-opencv libcurl4-openssl-dev libgtest-dev \
        cmake wget unzip curl git graphviz

RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ \
        tensorflow-gpu==1.7.0 keras==2.1.5 \
        numpy nose-timer requests Pillow graphviz


