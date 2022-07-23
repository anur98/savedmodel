# Activity Recognition #

Activity recognition involves predicting the movement of a person based on video data and traditionally involves deep domain expertise and methods from computer vision to correctly engineer features from the raw data in order to fit a machine learning model.
The aim of activity recognition is to recognize activity of a person from video sequences or still images.

Problem Statement: To detect activities performed by SLB workers while working in workshop using CCTV video feed.

## Approach 1: Activity Recognition using Image Classification ##
In this approach, we have used an up-stream task of person detection using YOLO model and a down-stream task of classification for the detected persons using an image classification model. This image classification model is built using transfer learning over pretrained [xception](https://keras.io/api/applications/xception/) base model.

Libraries Used
1)	Tensorflow 2.3.4
2)	opencv 4.6.0
3)	numpy 1.18.5
4)	Pillow  9.1.1
5)	scikit-learn 1.0.2
6)	matplotlib 3.5.2
7)	scipy 1.7.3


### Dataset Preparation for Approach 1: ###

The dataset for this task is an Image dataset (similar to public datasets like MNIST). 

Dataset can be prepared in one of the 2 ways:

**Creating Dataset (Method 1)**:
* Create a folder/directory with videos for creating dataset. Please find the videos which I have used for creating the datset in the given link: [Final Inspection Region Camera, RAK Location](https://storage.cloud.google.com/df-anuragshukla/Dataset_prep_videos/FIR_dataset_creation_videos.zip?authuser=2)
* Open the Jupyter Notebook : “ExtractingDataFromVideos.ipynb” 
* Set Configs (There is a configs cell in the Notebook):
  * `video_dir`: The directory which contains all the videos for dataset creation.
  * `output_dir`: The directory which will contain all the extracted images
  * `output_image_prefix`: Prefix for naming the extracted image
  * `stride`: Frame stride for extraction. Eg: If stride = 10, the frames extracted would be (0, 10, 20, … ). This is important because consecutive frames are very similar, hence you would waste a lot of time sorting through these frames if you don’t select an appropriate stride.
  * `yolo_model_path`: Path of person detection yolo model 
*	Run the Notebook
*	Sort the dataset into required classes

 
**Creating Dataset (Method 2):**
*	Take screenshots of the target video(s) to create images and store all the images in a folder. (Preferably using Snapshot functionality in VLC media player)
*	Open the Jupyter Notebook : “ExtractingDataFromImages.ipynb” 
*	Set Configs (There is a configs cell in the Notebook):
    * `image_dir`: The directory which contains all the images for dataset creation.
    * `output_dir`: The directory which will contain all the extracted images
    * `output_image_prefix`: Prefix of the name of each extracted image
    * `yolo_model_path`: Path of person detection yolo model
*	Run the Notebook
*	Sort the dataset into required classes

 
After extracting the dataset using one of the above methods, the folder structure of dataset should look like this:

Please find the datasets that I have created at the following link: 
*	[Final Inspection Region Model, Location RAK](https://storage.cloud.google.com/df-anuragshukla/Dataset/datasetFIR.zip?authuser=2)
*	[Teardown 1 and 2 model, Location KATY](https://storage.cloud.google.com/df-anuragshukla/Dataset/US_Dataset_12.zip?authuser=2)

### Model Training and Inferencing for Approach 1 ###

**Model Training Steps:**
*	Create a dataset using Dataset Preparation steps mentioned above (Or in the “Dataset Preparation.docx” document)
*	Open the Jupyter notebook : “TransferLearningTraining.ipynb” 
*	Set the configs: 
    * `train_dir`: The directory in which training data is present
    * `validation_dir`: The directory in which validation data is present
    * `test_dir`: The directory in which test data is present
    * `num_classes`: Number of classes in the dataset
    * `BATCH_SIZE`: batch size for training
    * `IMG_SIZE`: Input image dimensions for the model
    * `dense_layer_size`: size of dense layer after base model
    * `no_of_epochs` : No of epochs for training
    * `filepath`: Path for saving tuned model
    * `dropout_1`: Dropout after Global Avg Pooling layer. (Range: (0,1))
    * `dropout_2`: Dropout after the dense layer. (Range: (0,1))
    * `misclassified_folder`: Folder for storing misclassified test examples
* Run the Jupyter Notebook
* Your Model will be saved in the path corresponding to the ‘filepath’ configuration that you have specified. Please find the model that I have trained in the link given below:
*	[Final Inspection Region Model, Location RAK](https://storage.cloud.google.com/df-anuragshukla/Models/FIR_Model.zip?authuser=2)
*	[Teardown 1 and 2 model, Location KATY](https://storage.cloud.google.com/df-anuragshukla/Models/US_model12.zip?authuser=2)
 


**Inferencing**

There are 2 notebooks for inferencing on a model. One of the approaches involves smoothing of results.

Smoothing of results involves splitting the entire video into chunks of windows where each window consists of 50 consecutive frames and each frame in the window is labelled as the mode prediction label.

Inferencing Steps:
*	Open the Jupyter notebook : “Activity_Recognition_Inference.ipynb” or “Activity_Recognition_Inference_with_smoothing.ipynb”
*	Set the configs: 
    *	`scaling factor`: The scaling factor for images used while training (Eg , scaling factor = 1./255.)
    *	`video_path`: Path of video for inference
    *	`IMG_SIZE`: Size of the input layer of model
    *	`Model_path`: Path of the saved classification model
    *	`Yolo_model_path`: Path of person detection yolo model
    *	`output_video_path`: Path where inference video needs to be created (including name of the video in path)
    *	 `labels`: A list of labels for the current classification model. (Eg. labels = ["NotWorkingOnTool", "WorkingOnTool"])
    *	`colors`: A list of color codes for bounding boxes, corresponding to the labels in BGR format. (Eg. colors = [(0, 0, 255), (0,255,0)])
    *	`Class_mode`: ‘binary’ / ‘categorical’
*	 Run the notebook
*	Inference video can be found at the path specified by  ‘output_video_path’ config.

Please find the inference videos generated by me at:  [Final Inspection Region Camera, Location RAK Inferences](https://storage.cloud.google.com/df-anuragshukla/Inference%20Videos/FIR_RAK_Inferences.zip?authuser=2)
 

**Inferencing for KATY location (US Region):**

The YOLO model that we used for detecting persons wasn’t able to detect sitting persons (which was a class for KATY region). Hence, we used a separate YOLO model for detecting sitting people. This model was used to quickly build proof of concept for US region. The recommended way for activity recognition is to retrain the YOLO model so that a single YOLO model is able to detect sitting as well as standing persons.

The pipeline for inferencing with is given below:
 
Inference Steps:
*	Open the Jupyter notebook : “Activity_Recognition_Inferencing_US_Pipeline.ipynb” 
*	Set the configs: 
    *	`scaling factor`: The scaling factor for images used while training (Eg , scaling factor = 1./255.)
    *	`video_path`: Path of video for inference
    *	`IMG_SIZE`: Size of the input layer of model
    *	`classification_model_path`: Path of the saved classification model
    *	`yolo_person_model_path`: Path of person detection yolo model
    *	`yolo_sitting_model_path`: Path of sitting-person detection yolo model
    *	`output_video_path`: Path where inference video needs to be created (including name of the video in path)
    *	 `labels`: A list of labels for the current classification model. (Eg. labels = ["NotWorkingOnTool", "WorkingOnTool"])
    *	`colors`: A list of color codes for bounding boxes, corresponding to the labels in BGR format. (Eg. colors = [(0, 0, 255), (0,255,0)])
    *	`class_mode`: ‘binary’ / ‘categorical’
*	Run the notebook
*	Inference video can be found at the path specified by  ‘output_video_path’ config.
Please find the inference videos generated by me at:  [Teardown 1 and 2 camera, Location KATY(Inferences)](https://storage.cloud.google.com/df-anuragshukla/Inference%20Videos/Katy%20US%20Inferences.zip?authuser=2).



## Approach 2: Activity Recognition using MoViNet ##
In this approach, we have fine-tuned the MoViNet model for activity recognition. MoViNet is an activity recognition model which is trained on Kinetics 600 activity recognition dataset. The Kinetics dataset about 500,000 videos belonging to 600 classes. We have downloaded the weights of pre-trained MoVinet from tensorflow hub and fine-tuned it for our use case. 

### Dataset Preparation for Approach 2: ###

To extract video dataset from CCTV videos, we take a window of 50 frames and extract people from these frames. Then we make a video for every unique person seen in the frame to get video data. Finally, we sort these videos according to activities.

Steps to create the dataset:
*	Create a folder/directory with videos for creating dataset. Please find the videos that I have used to create the dataset here: [Final Inspection Region Videos, Location RAK](https://storage.cloud.google.com/df-anuragshukla/Dataset_prep_videos/FIR_dataset_creation_videos.zip?authuser=2)
*	Open the Jupyter Notebook: “Extracting_Video_Dataset.ipynb”
*	Set Configs:
    *	`video_dir`: The directory which contains all the videos for dataset creation.
    *	`output_dir`: The directory which will contain all the extracted videos
    *	`output_video_prefix`: Prefix of the naming the extracted videos.
    *	`yolo_model_path`: Path of the person detection YOLO model
*	Run the Notebook
*	Sort the dataset into required classes
*	Split the dataset into Train, test and validation sets and put these sets in different folders

 
After extracting the dataset, the folder structure of the dataset should look like this:
 
The dataset prepared by me for this task can be found at the following link: [Video dataset for Final Inspection Region, Location RAK](https://storage.cloud.google.com/df-anuragshukla/Dataset/Dataset_movinet%202.zip?authuser=2)

### Model Training and Inferencing for Approach 2 ###

For training the model, we have taken the weights of MoViNet classifier from tensorflow hub. Then we have used the Schlumberger dataset for fine-tuning it.

**Model Training Steps:**
*	Create a dataset using Dataset Preparation steps mentioned above (Or in the “Dataset Preparation.docx” document)
*	Open the Jupyter notebook : “MovinetTraining.ipynb”
*	Set the configs:
    *	`num_classes`: Number of classes in your dataset
    *	`filepath`: Path for saving trained model
    *	`num_epochs`: number of epochs for training.
*	Run the Jupyter Notebook
*	Your Model will be saved in the path corresponding to the ‘filepath’ configuration that you have specified. The model trained by me can be found at: [Movinet Model Final Inspection Region, Location RAK](https://storage.cloud.google.com/df-anuragshukla/Models/RAK_Movinet_Model.zip?authuser=2)
 

**Inferencing**

*	Open the Jupyter notebook : “MovinetInferencing.ipynb”
*	Set the configs: 
    *	`yolo_model_path`: Path of person detection yolo model
    *	`classes`: A list of class labels for the current classification model. (Eg. labels = ["NotWorkingOnTool", "WorkingOnTool"])
    *	`class_colors`: A list of color codes for bounding boxes, corresponding to the labels in BGR format. (Eg. colors = [(0, 0, 255), (0,255,0)])
    *	`video_path`: Path of video for inference
    *	`model_path`: Path of the saved classification model
    *	`output_video_path`: Path where inference video needs to be created (including name of the video in path)
* Run the Notebook
* Inference video can be found at the path specified by   ‘output_video_path’ config. The inference videos created by me can be found at: [Final Inspection Region Camera, Location RAK(Inferences)](https://storage.cloud.google.com/df-anuragshukla/Inference%20Videos/MoviNetFIRnewInferences.zip?authuser=2)

 
