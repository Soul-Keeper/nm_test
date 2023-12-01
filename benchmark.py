import os
import cv2
import time
import numpy as np

from tqdm import tqdm
from ultralytics import YOLO
from deepsparse import Pipeline
from deepsparse.benchmark.benchmark_model import benchmark_model


images_path = "data/images/test2007"
filenames = next(os.walk(images_path), (None, None, []))[2][:300]

results = []

models_dict = {
    "YOLOv8-nano": 'yolov8n.pt',
    "YOLOv8-small": 'yolov8s.pt',
    "YOLOv8-medium": 'yolov8m.pt'
}

for model_name in models_dict:
    model = YOLO(models_dict[model_name])

    for file in tqdm(filenames, desc="{}".format(model_name)):
        image = cv2.imread(images_path + '/' + file)

        eval_time = []
        st = time.time()
        pred = model(source=image, verbose=False)
        et = time.time()
        eval_time.append(et - st)

    results.append("{}: {}".format(model_name, round(sum(eval_time)/len(eval_time), 3)))

sparse_models_dict = {
    "YOLOv8-nano": 'zoo:yolov8-n-coco-base',
    "YOLOv8-nano-pruned": 'zoo:yolov8-n-coco-pruned49',
    "YOLOv8-nano-pruned-quant": 'zoo:yolov8-n-coco-pruned49_quantized',
    "YOLOv8-small": 'zoo:yolov8-s-coco-base',
    "YOLOv8-small-pruned": 'zoo:yolov8-s-coco-pruned55',
    "YOLOv8-small-pruned-quant": 'zoo:yolov8-s-coco-pruned55_quantized',
    "YOLOv8-medium": 'zoo:yolov8-m-coco-base',
    "YOLOv8-medium-pruned": 'zoo:yolov8-m-coco-pruned80',
    "YOLOv8-medium-pruned-quant": 'zoo:yolov8-m-coco-pruned80_quantized',
}

for model_name in sparse_models_dict:

    yolo_pipeline = Pipeline.create(
        task="yolo",
        model_path=sparse_models_dict[model_name],
    )

    for file in tqdm(filenames, desc="{}".format(model_name)):
        image = cv2.imread(images_path + '/' + file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640), interpolation = cv2.INTER_AREA)

        eval_time = []
        st = time.time()
        pred =  yolo_pipeline(images=image, iou_thres=0.7, conf_thres=0.25)
        et = time.time()
        eval_time.append(et - st)

    results.append("{}: {}".format(model_name, round(sum(eval_time)/len(eval_time), 3)))

for rec in results:
    print(rec)

results = benchmark_model('zoo:yolov8-m-coco-pruned80_quantized')
print(results)

results = benchmark_model('zoo:yolov8-m-coco-pruned80_quantized', scenario="async")
print(results)

results = benchmark_model('zoo:yolov8-m-coco-pruned80_quantized', batch_size=32)
print(results)

results = benchmark_model('zoo:yolov8-m-coco-pruned80_quantized', scenario="async", batch_size=32)
print(results)
    