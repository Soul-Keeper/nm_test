FROM neuralmagic/deepsparse-server:latest

RUN apt update && apt upgrade -y

WORKDIR /usr/src/app
COPY . /usr/src/app/

RUN pip install -r requirements.txt
RUN pip install -U ultralytics