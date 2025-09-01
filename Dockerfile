
# Use newer PyTorch + CUDA + Ubuntu

FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel


# Install OS dependencies
RUN apt-get update -y && apt-get install -y \
  libglib2.0-dev libsm6 libxext6 libxrender-dev freeglut3-dev ffmpeg wget gnupg2

# Python packages
RUN pip install --no-cache-dir \
  gym-super-mario-bros==7.3.2 \
  opencv-python==4.7.0.72 \
  future==0.18.3 \
  pyglet==1.5.27

# Set workdir
WORKDIR /Super-mario-bros-PPO-pytorch

