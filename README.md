# Dhaka AI

My solution of [Dhaka Ai Traffic Detection Challenge](https://dhaka-ai.com/).

> An international AI-based Dhaka Traffic Detection Challenge funded by Elsevier would be co-organized during STI 2020

## Dataset
 The dataset is composed of vehicle images, where an image contains a vehicle of one or more of 21 different classes of vehicle. This makes the dataset useful for multiple vehicle detection and recognition. The considered vehicle classes are: ambulance, auto-rickshaw, bicycle, bus, car, garbage van, human hauler, minibus, minivan, motorbike, Pickup, army vehicle, police car, rickshaw, scooter, Suv, taxi, three-wheelers (CNG), truck, van, wheelbarrow. 


### Inference on test image
<img src="img/inference-1.jpg" alt="Inference on test image" width="45%"/> <img src="img/inference-2.jpg" alt="Inference on test image" width="45%"/>




### Train data overview
<img src="YOLOv5/runs/train/exp8/labels.png" alt="Train Data Overview" width="60%"/>

### Train result
<img src="YOLOv5/runs/train/exp8/results.png" alt="Train Result" width="85%"/>


### Validation batch predictions
`test_batch[0-9]+_pred.jpg` shows—

<img src="YOLOv5/runs/train/exp8/test_batch0_pred.jpg" alt="validation batch predictions" width="45%"/> <img src="YOLOv5/runs/train/exp8/test_batch1_pred.jpg" alt="validation batch predictions" width="45%"/>


### Validation batch label
`test_batch[0-9]+_labels.jpg` shows—

<img src="YOLOv5/runs/train/exp8/test_batch0_labels.jpg" alt="validation batch labels" width="45%"/> <img src="YOLOv5/runs/train/exp8/test_batch1_labels.jpg" alt="validation batch labels" width="45%"/>


### Train batch mosaics and labels:
`train_batch[0-9]+.jpg` shows—

<img src="YOLOv5/runs/train/exp8/train_batch0.jpg" alt="train batch mosaics and labels" width="45%"/> <img src="YOLOv5/runs/train/exp8/train_batch2.jpg" alt="train batch mosaics and labels" width="45%"/>



### Best Result
MBSTU_Underrated: 0.1346


