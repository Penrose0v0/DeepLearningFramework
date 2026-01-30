FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# zsh
RUN apt-get update && apt-get install -y wget git zsh curl
SHELL ["/bin/zsh", "-c"]
RUN wget http://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh
RUN sed -i "s/# zstyle ':omz:update' mode disabled/zstyle ':omz:update' mode disabled/g" ~/.zshrc

# python (3.12)
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.12 python3.12-dev python3.12-venv python3-pip
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# venv
RUN python -m venv /root/venv/work
RUN source /root/venv/work/bin/activate

# utils
RUN apt-get update && apt-get install -y htop vim ffmpeg

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /root
CMD ["zsh"]