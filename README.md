# Car_detection_model.
Train yolo model on training images of road recorded video in which cars are riding. After this also evaluate this model. 

# Car Object Detection using YOLOv8

A computer vision project for detecting cars in road scenes using
YOLOv8 object detection.

The main focus of this project was not only training a detection model,
but also understanding how dataset splitting affects model
generalization when the dataset consists of video frames.

---

## Dataset

This project uses the **Car Object Detection** dataset from Kaggle.

The dataset contains road images extracted from videos along with
bounding-box annotations stored in a CSV file.

Dataset:
https://www.kaggle.com/datasets/kaizen19/car-object-detection

### Important Dataset Observation

The training images follow a naming pattern such as:

text
vid_4_1000.jpg
vid_4_1020.jpg
vid_4_1040.jpg
...

The numeric part of the filename represents the frame position in the video sequence.

Therefore, consecutive images can contain highly similar visual information.


---

### Project Approach

I experimented with different dataset-splitting and training strategies to investigate overfitting and improve model generalization.

The project contains four main notebooks:

notebooks/
│
├── preprocessing.ipynb
├── Model.ipynb
├── preprocessing1.ipynb
└── Model1.ipynb


---

#### Experiment 1 — Random Train/Validation Split

Notebook

preprocessing.ipynb

The original CSV file contains the image names and bounding-box coordinates:

image
xmin
ymin
xmax
ymax

Preprocessing

The following steps were performed:

1. Load the CSV annotations.


2. Group bounding boxes by image.


3. Convert the bounding boxes into YOLO annotation format.


4. Create a labels directory containing .txt files.


5. Split the images into training and validation sets using a random 80/20 split.


6. Create the YOLO dataset structure.



dataset/
│
├── images/
│   ├── train/
│   └── val/
│
└── labels/
    ├── train/
    └── val/

A data.yaml file was also created for YOLO training.


---

#### Experiment 1 — Model Training

Notebook

Model.ipynb

The model used was:

YOLOv8n

Training configuration included:

Epochs: 50
Image size: 640
Batch size: 16

The training and validation losses were plotted to analyse the model's behaviour.

##### Problem Observed

The model showed signs of overfitting.

The model was trained on a relatively small number of annotated images, and the images came from video sequences.

This raised an important question:

> Is a random train/validation split appropriate for a dataset made from video frames?



This led to the second experiment.


---

#### Experiment 2 — Chronological Train/Validation Split

Notebook

preprocessing1.ipynb

Instead of randomly splitting the images, the image filenames were analysed.

For example:

vid_4_1000.jpg
vid_4_10020.jpg
vid_4_10040.jpg
vid_4_10060.jpg
...

The numeric frame ID was extracted from the filename.

vid_4_1000.jpg → 1000
vid_4_1020.jpg → 1020
vid_4_1040.jpg → 1040

The dataset was then sorted according to this frame ID.

Images
   ↓
Extract frame ID
   ↓
Sort chronologically
   ↓
80/20 split without shuffling

The split was performed with:

train_test_split(
    unique_image,
    test_size=0.2,
    shuffle=False
)

This preserves the temporal order of the frames.


---

Why Change the Split?

When video frames are randomly split, frames from very similar parts of the same sequence can appear in both training and validation sets.

This can make validation performance less representative of how the model behaves on a genuinely different part of a video.

Therefore, chronological splitting was tested as a more meaningful evaluation strategy for this dataset.


---

#### Experiment 2 — Model Training

Notebook

Model1.ipynb

Again, YOLOv8n was used.

The training configuration included:

Epochs: 50
Image size: 640
Batch size: 16
Weight decay: 0.001
Mosaic: 1.0
Scale: 0.5
Translate: 0.2
Horizontal flip: 0.5
Close mosaic: 10
Patience: 10
Seed: 42

These settings were used to improve the model's ability to generalize.


---

# Results

The final experiment achieved:

Peak Validation mAP@0.5 : 0.99454
Final Validation mAP@0.5: 0.99379

The training and validation box loss and classification loss were also visualized to analyse the model's learning behaviour.

The final experiment showed a much smaller separation between training and validation behaviour compared with the earlier experiments.


---

# Experiment Comparison

Experiment	Split Strategy	Training Strategy	Result

1	Random 80/20 split	Basic YOLOv8n training	Overfitting observed
2	Chronological 80/20 split	YOLOv8n + augmentation/regularization	Better generalization
3	Final tuned experiment	YOLOv8n + stronger augmentation/training settings	Peak mAP@0.5 = 0.99454



---

# Testing

The trained model was also tested on the separate testing images.

The best trained weights were loaded:

best.pt

Predictions were generated with a confidence threshold of:

0.25

The predicted bounding boxes were visualized on the test images.

The model was able to detect different numbers of cars in the test images depending on the scene.


---

# Key Learnings

This project helped me understand that model performance depends not only on the model architecture, but also heavily on the dataset and evaluation strategy.

1. Dataset structure matters

Images that look like independent samples may actually be consecutive frames from the same video.

2. Random splitting is not always appropriate

For video-frame datasets, random splitting can place visually similar frames in both training and validation sets.

3. Overfitting can come from the dataset

With limited training data and similar backgrounds, the model can learn scene-specific visual patterns instead of learning features that generalize well.

4. Data augmentation can help

Augmentation was introduced to expose the model to more variation during training and improve generalization.

5. Loss curves are useful

Training and validation loss curves were used to understand whether the model was learning effectively or showing signs of overfitting.

6. Validation performance needs context

A high validation score does not automatically mean that a model will generalize well to completely different videos or environments.


---

Tech Stack

Python

YOLOv8

Ultralytics

OpenCV

Pandas

NumPy

Matplotlib

Scikit-learn

Jupyter Notebook / Google Colab



---

# Project Structure

car-object-detection/
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── Model.ipynb
│   ├── preprocessing1.ipynb
│   └── Model1.ipynb
│
├── src/
│   ├── preprocessor.py
│   ├── predictor.py
│   └── main.py
│
├── data.yaml
├── requirements.txt
├── README.md
└── .gitignore

The notebooks contain the experimentation and model-development process, while the src/ directory contains the application/inference code.


---

# Future Work

Evaluate the model on completely unseen road videos.

Increase the diversity of training data.

Add more independent video sequences.

Further investigate overfitting.

Compare different YOLO model sizes.

Measure inference speed for real-time detection.

Integrate the trained model into a real-time road-video detection application.
