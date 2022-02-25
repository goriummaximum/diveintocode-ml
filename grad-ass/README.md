#  Traffic sign detection on real Vietnamese streets

## What?
My graduation assignment of class ML2010 of Dive Into Code 2021 - 2022. The project is the effort to train the computer to be able to detect traffic sign on Vietnamese street via images (done), video (in the future), webcam (in the future), by using YOLOv4 with the implementation of Darknet proposed by AlexeyAB and Pjreddie. In addition, thanks to the existed well-made traffic sign dataset from Zalo AI challenge 2020, the training process is much easier and faster, and the final mAP@0.5 is 73.64% on self-made validation set, the speed of ~13FPS on Nvidia Tesla T4.

## Why?
To graduate from the course. To help drivers to detect traffic sign, protect themselves from accidents, danger, and police... as well as the traffic in a whole from unintentional loss of focus on a long run ride, which makes them miss the traffic signs ahead, ...and myself...

## How?
Using existed ML model YOLOv4 - darknet, and existed dataset from Zalo AI challenge 2020.  
Google Colab as the main processing environment, mostly use Nvidia Tesla T4 GPU for training and testing.

## Examples of predicted images
Varieties of traffic sign
![1250](https://github.com/goriummaximum/diveintocode-ml/blob/master/grad-ass/examples/1250.jpg)
Detect a traffic sign faraway
![12023](https://github.com/goriummaximum/diveintocode-ml/blob/master/grad-ass/examples/12023.jpg)

## File system
`data_analysis.ipynb` import dataset and do analysis, display some raw images with bboxes.  
`prepare_data.ipynb` preprocessing, train test split, augmentate dataset and generate YOLO data format.  
`prepare_model.ipynb` download Darknet and YOLOv4 related files, configure YOLOv4, make model to run with GPU.  
`train.ipynb` train model.  
`test.ipynb` predict images, calculate mAP, predict video(coming soon).  
`helper_code/` include common code used intensively by above files.  
`data/` anything related to dataset.  
`model/` anything related to model.  

## How to run?
1. download dataset (.zip) [here](https://drive.google.com/file/d/17BFYn077nh9wuhwnErg6GOPKVgPWY9Nx/view?usp=sharing).  
2. copy the downloaded .zip to data folder and extract.  
3. go to `helper_code/common.py` and change the `DATA_SRC` to  `${your_absolute_path_to_dataset}/za_traffic_2020/traffic_train`.    
4. also change the `MODEL_SRC` to `${your_absolute_path_to_repo}/model`.  
5. run all code blocks in `data_analysis.ipynb`.  
6. run all code blocks in `prepate_data.ipynb`.  
8. run all code blocks in `prepare_model.ipynb`.  
9. copy `model/cfg/yolov4-custom.cfg` to `model/darknet/cfg`.  
10. (optional) run all code blocks in `train.ipynb` if you want to train again.  
11. download the weights file (.weights)  [here](https://drive.google.com/file/d/1Ov8lZA5FLlnqZWKfnuiHUVxreLIMh0n3/view?usp=sharing).  
12. copy the the downloaded `yolov4-custom_8100.weights` to `model/darknet/backup`.  
13. open `test.ipynb` change the `backup/yolov4-custom_last.weights` to `backup/yolov4-custom_8100.weights` in every code block that exists.  
14. run all code blocks in `test.ipynb` for testing.  
15. if you want to predict custom images, go to `predict an image` code block in `test.ipynb` and change the path to your image.  
### Note:  
please run the first code block exists in every `.ipynb` file at least 1 time to install essential libraries.  

## Others
Please read `mef2010_HoHoangThien_Long.pdf` as a report for more explaination.  


