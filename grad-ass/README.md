#  Traffic sign detection on real Vietnamese streets

## What?
My graduation assignment of class ML2010 of Dive Into Code 2021 - 2022. The project is the effort to train the computer to be able to detect traffic sign on Vietnamese street via images (done), video (in the future), webcam (in the future), by using YOLOv4 with the implementation of Darknet proposed by AlexeyAB and Pjreddie. In addition, thanks to the existed well-done traffic sign dataset from Zalo AI challenge 2020, the training process is much easier and faster, and the final mAP@0.5 is 73.64% on self-made validation set, the speed of ~13FPS on Nvidia Tesla T4.

## Why?
To graduate from the course. To help drivers to detect traffic sign, protect themselves from accidents, danger, and police... as well as the traffic in a whole from unintentional loss of focus on a long run ride, which makes them miss the traffic signs ahead, ...and myself...

## How?
Using existed ML model YOLOv4 - darknet, and existed dataset from Zalo AI challenge 2020.  
Google Colab as the main processing environment, mostly use Nvidia Tesla T4 GPU for training and testing.

## File system
`data_analysis.ipynb` import dataset and do analysis, display some raw images with bboxes.  
`prepare_data.ipynb` preprocessing, train test split, augmentate dataset and generate YOLO data format.  
`prepare_model.ipynb` download Darknet and YOLOv4 related files, configure YOLOv4, make model to run with GPU.  
`train.ipynb` train model.  
`test.ipynb` predict images, calculate mAP, predict video(coming soon).  
`helper_code/` include common code using intensively by above files.  
`data/` anything related to dataset.  
`model/` anything related to model.  

## How to run?
```
bash ./run.sh
```

## Others
Please read `mef2021_HoHoangThien_Long.pdf` as a report for more explaination.  


