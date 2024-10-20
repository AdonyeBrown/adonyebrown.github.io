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

```bash
pip install detectron2
```
### Load a Pre-trained Model

Detectron 2 provides pre-trained models for various tasks, making it easy to get started. Here's how to load a pre-trained model:

```
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

# Load a pre-trained model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
```

### Running Inference
Once the model is loaded, you can run inference on an image:

```
from PIL import Image
import numpy as np

# Load an image
image = Image.open("path/to/image.jpg")
image = np.array(image)

# Run inference
outputs = predictor(image)
print(outputs)
```

### Visualizing Results with TensorBoard
TensorBoard is a great tool for visualizing the training process and results of your model.

### Setting Up TensorBoard

```
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

# Set up logger
setup_logger()

# Set up configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "path/to/weights.pth"

# Set up TensorBoard
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 30000
cfg.SOLVER.STEPS = (20000, 25000)
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.TEST.EVAL_PERIOD = 1000
cfg.OUTPUT_DIR = "path/to/output"

# Start training
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
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
- F1 Score: The harmonic mean of precision and recall, providing a balanced measure of the model's performance.

**Conclusion**
Detectron 2 is a powerful tool for object detection and segmentation, providing a robust framework for various computer vision tasks. By understanding and utilizing the appropriate metrics, you can effectively evaluate and improve your model's performance.

