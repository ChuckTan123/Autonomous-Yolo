# Final Project: Cat Detection, Yolo v1 mxnet implementation
    
    Paper link: https://arxiv.org/abs/1506.02640
    
## Goal: The industry applications of object detection are tremendous. Deep Learning based object detection algorithm has proven to be promising. This project is targeting to train students' capabilities to do a fully deep learning project, which includes below aspects:

    1. Having basic machine learning ethics by be able to answer these questions:
        (1) What is the type of this project about? (Supervised, Unsupervised, Reinforcement Learning)
        (2) What is a valid machine results? 
        (3) What is the ratio should the dataset be split?
        (4) How should the dataset be split?
        (5) How much data is sufficient/necessary for this project?
        (6) If it is not sufficient, where and how can students get the data?
        (7) How much time/effort/money do students need to get the data? Is it worthy? Any business value bigger than the effort?
        (8) What is the precision & recall? What does it mean specifically in object detection? 
        ...
        
    2. Being able to read & understand the algorithm in a deep learning paper (here the Yolo paper is our target) by be able to anwser these questions below:
        (1) What is the data format? (image based, or others)
        (2) What is the target label format? (how to define the a box)
        (3) What is the main idea of the algorithm?
            I, What is the feature extraction componet of the algorithm?
            II, How does the algorithm map the prediction to target label?
            III, What is the detailed loss function?
            IV, What does the final 7x7 grid mean, and how it can interpreted for detection purpose
            V, What is the training details in this algorithm?
       
    3. Being able to build the data io, training&testing process based on a deep learning paper in MXNET.
        (1) What is the data io in mxnet (.rec)?
        (2) How to create the customized dataset of a certain raw dataset. 
        (3) How to build the body of the network architecture
        (4) Loss function design
        (5) Training process design 
        (6) Wild Testing process design
        (7) How to productize the trained model to certain business
        
## Project Definition

### Track 1 (easy): build an inference module based on the model provided by the teacher.

    MVP(Minimum Viable Product):
        1. An inference script, which can load and run forward inference to draw boxes on the image containing cats.
    
    Hardware Requirment:
        Any CPU machine 

### Track 2 (medium): build a training process module based on the model provided by the teacher.

    MVP(Minimum Viable Product):
        1. A script to convert raw image data to MXNET .rec file data
        2. Create training dataset and validation dataset
        3. A scirpt of network architecture based on the understanding of MXNET, including:
            (1) CNN body
            (2) Loss function 
        4. A script of training process
        
    Hardware Requirment:
        Any CPU machine         

### Track 3 (Hard): build&run an training module.

    MVP(Minimum Viable Product):
        1. A script to convert raw image data to MXNET .rec file data
        2. Create a full dataset AND a very small dataset (~10 images)
        3. A scirpt of network architecture based on the understanding of MXNET, including:
            (1) CNN body
            (2) Loss function 
        4. A script of training process
        5. Train the network with the very small dataset (~10 images) to overfitting. The studnets should observe the perfect bounding box on the trained dataset.
           Please provide a ipynb file to draw the boxes on images
        
    Hardware Requirment:
        GPU machine with >6 GB memory (Dont have GPU? Review mxnet-week2)
    
    
### Track 4 (Hell): Deliver an usable cat detection model from scratch
    MVP(Minimum Viable Product):
        1. A script to convert raw image data to MXNET .rec file data
        2. Create a full dataset AND a very small dataset (~10 images)
        3. A scirpt of network architecture based on the understanding of MXNET, including:
            (1) CNN body
            (2) Loss function 
        4. A script of training process
        5. Train the network with the very small dataset (~10 images) to overfitting. The studnets should observe the perfect bounding box on the trained dataset.
           Please provide a ipynb file to draw the boxes on images
        6. Train the network to converge
           Please report the training&validation loss&accuracy
        7. Build the inference script and do wild testing on random google images contaning cats in
        
    Hardware Requirment:
        GPU machine with >6 GB memory (Dont have GPU? Review mxnet-week2)
        
### To start the project, download the data/model/pretrained files from google drive. Unzip them in corresponding folder.                   