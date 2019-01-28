#Ubuntu image
FROM ubuntu:16.04

# FROM defines the base image
FROM nvidia/cuda:9.0-base
FROM nvidia/cuda:9.0-cudnn7-runtime
FROM nvidia/cuda:9.0-cudnn7-devel

#set the working directory
WORKDIR /dasiamrpn

RUN apt-get update && apt-get -y install python-pip python-dev 
RUN pip install pip -U
RUN pip install zmq requests Pillow

ADD run_install.sh /dasiamrpn
ADD code /dasiamrpn/code

#RUN /bin/bash ./run_install.sh

RUN apt-get update && apt-get -y install libgtk2.0-dev
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install opencv-python
RUN pip install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl

WORKDIR /dasiamrpn/code

CMD python tracker_zmq.py
