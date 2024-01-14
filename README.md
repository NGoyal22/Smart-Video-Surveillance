# SmartSurveillanceSystem
![Smart Surveillance System](./images/project.jpeg)


## Abstract
Object movement identification is one of the most researched problems in the field of computer vision. In this task, we try to classify a pixel as foreground or background. Even though numerous traditional machine learning and deep learning methods already exist for this problem, the two major issues with most of them are the need for large amounts of ground truth data and their inferior performance on unseen videos. Since every pixel of every frame has to be labeled, acquiring large amounts of data for these techniques gets rather expensive. Recently,  Zhao et al. proposed one of a kind Arithmetic Distribution Neural Network (ADNN) for universal background subtraction which utilizes probability information from the histogram of temporal pixels and achieves promising results. Building onto this work, we propose an intelligent video surveillance system that will use ADNN architecture for motion detection with some additional features like anomaly detection built within it.

## Running environment 
`Anaconda + Pytorch: conda 4.12.0` `python 3.7.13` `CUDA 10.1` `torch 1.7.1` `conda activate adCVPR18`

## AnomalyDetectionCVPR2018-Pytorch
* C3D feature extraction

## 
C3D Weights
I couldn't upload here the weights for the C3D model because the file is too big, but it can be found here:
https://github.com/DavideA/c3d-pytorch


## Pre-Trained Anomaly Detector
Check out <a href="exps/c3d/">exps/</a> for for trained models on the pre-computed features

## How to run the code:

We have provided a fully fledged code and we need to run below steps in order to get results:

Step1. First we need to run `extract_frames_ft.py` and after this we get input frames corresponding to its ground truth which will save to Input_data/video1. 

Step2. Then we run `adnn_detect_work.py` ADNNet (Arithmetic Distribution Neural Network) for background subtraction provided a pre-trained model which is trained with less than 1\% of ground truth frames from the CDNet2014 dataset and save the output frames in adnn_output folder. 

Step3. Then we run `bayesian_refine_work.py` and will save our final refined frames in the bayesian_output folder.  

Step4. To extract trimmed video we run `generate_trimmed_video.py`. This will save trimmed video in new folder named 'trimmed_video'. We have set different threhold values, depending on the input video. It will provide us the trimmed input video containing input frames having some motion in it and exclude frames which are still and not having any motion. By extracting trimmed video we are saving lot of time in anamoly detection. 

Step5. In this step we are extracting the features of our trimmed video. For that we need to run `feature_extractor.py` and this will save the features in feature_output folder. You can have a look into this folder and find the .txt file containing the features of the trimmed video. We use this feature to find the anamoly score which is our final step. 

Step6. Finally, we have extracted features from trimmed video and we use these features to find the anamoly score. We run `generate_anomaly_score.py` and it will save the graph of anamoly scores in anamoly_score folder. 

        
By just following above steps we will get results. To summerize, we first have an input image and we are finding the binary mask input frames with help of ADNN and then we use bayesian refine for the refinement of the input frames. Then we use refined frames from bayesian refine in order to get trimmed video with the help of thresholding technique. Then we will find features of trimmed input video and finally we calculate anamoly scores with the help of these features. 
