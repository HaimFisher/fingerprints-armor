FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    libsqlite3-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

# install python
RUN apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils python3-pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN python3 --version
RUN pip3 install -U setuptools

# install python 3rd party modules
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# remove unnecessary files
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# set the work directory
WORKDIR /workspace

# create and switch to a non-root user
RUN groupadd -r nusergroup && useradd -r -g nusergroup nuser
RUN chown -R nuser:nusergroup /workspace
RUN mkdir -p /home/nuser && touch /home/nuser/.bashrc
RUN echo 'export PS1="\\u:\\w$ "' >> /home/nuser/.bashrc
USER nuser
