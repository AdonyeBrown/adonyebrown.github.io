---
layout: post
title: Computer Vision
subtitle: Object Detection with Dectron2
category: [Important Features, architecture]
tags: [pytorch, detectron2_demo ,MLlib]
---

## Object detection with detectron2

## Introduction

Object detection is a fascinating field in computer vision that involves identifying and locating objects within an image or video. Unlike image classification, which simply assigns a label to an image, object detection not only classifies objects but also specifies their positions by drawing bounding boxes around them.

The Main Goals of Object Detection
- Classify objects present in the image.

- Locate those objects with precise coordinates.

Object detection models like YOLO (You Only Look Once), Faster R-CNN, and SSD (Single Shot MultiBox Detector) are commonly used due to their ability to perform real-time detection and maintain high accuracy. Among these, PyTorch's Detectron 2 has emerged as a powerful tool for this purpose


### Why Detectron2?

Detectron 2, developed by Facebook AI Research (FAIR), is an open-source library built on the PyTorch framework. It provides a comprehensive set of tools and algorithms for various computer vision tasks, making it a versatile choice for developers and researchers alike.



## Getting Started with Detectron 2
### Installation
First, you'll need to install Detectron 2 and its dependencies

```colab_notebook
!python -m pip install pyyaml==5.1
import sys, os, distutils.core
# Note: This is a faster way to install detectron2 in Colab, but it does not include all functionalities (e.g. compiled operators).
!git clone 'https://github.com/facebookresearch/detectron2'
dist = distutils.core.run_setup("./detectron2/setup.py")
!python -m pip install {' '.join([f"'{x}'" for x in dist.install_requires])}
sys.path.insert(0, os.path.abspath('./detectron2'))

```
### Colab provides you with free GPU
```
import torch, detectron2
!nvcc --version
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)
```
### Load a Pre-trained Model
```
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
```
First, download an image 
```
im = cv2.imread("/content/Igbo-people-and-their-cultural-heritage.jpg")
cv2_imshow(im)
```
Detectron 2 provides pre-trained models for various tasks, making it easy to get started. Here's how to load a pre-trained model:

```
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)
```

Look at the output
```
# look at the outputs.
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)
```

### Visualizing Results with TensorBoard

```
# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2_imshow(out.get_image()[:, :, ::-1])
```


### Metrics for Judging the Model
**Precision and Recall**
- Precision: Measures the accuracy of the positive predictions.

- Recall: Measures the ability of the model to find all relevant instances.

**Mean Average Precision (mAP)**
- mAP: A common metric for evaluating object detection models, which combines precision and recall across different thresholds.

**Intersection over Union (IoU)**
- IoU: Measures the overlap between the predicted bounding box and the ground truth bounding box.

**F1 Score**
- F1 Score: The harmonic mean of precision and recall, providing a balanced measure of the model's performance

**Conclusion**
Detectron 2 is a powerful tool for object detection and segmentation, providing a robust framework for various computer vision tasks. By understanding and utilizing the appropriate metrics, you can effectively evaluate and improve your model's performance.

[Click here](https://github.com/AdonyeBrown/detectron2_demo)
