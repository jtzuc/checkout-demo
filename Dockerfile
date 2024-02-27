FROM python:3.10-bullseye
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install --upgrade pip==21.1.1 && pip3 install -r requirements.txt
ENV DISPLAY=0
    