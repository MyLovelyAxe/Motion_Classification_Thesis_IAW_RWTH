# Classification of Human Body Activities

#### -- Project Status: [Active]

## Project Introduction

The purpose of this project is to classify human body activity with machine learning methods based on joint coordinates from [Captury Live](https://captury.com/captury-live/).
This thesis theme is offered by M.Sc Apostolos Vrontos from [Institute of Industrial Enginering and Ergonomics (IAW)](https://www.iaw.rwth-aachen.de/go/id/ieplw/?lidx=1) at [RWTH Aachen University](https://www.rwth-aachen.de/go/id/a/?lidx=1), and the codes are created by Jialei Li, from major [Robotic System Engineering](https://www.academy.rwth-aachen.de/en/programs/masters-degree-programs/detail/msc-robotic-systems-engineering), RWTH Aachen University.

### Methods Used

* Spacial coordinates of body joints
* Feature selection
* Statistic
* Data Visualization
* Machine Learning
* Classification Modelling

### Technologies

* Python
* Pandas
* Matplotlib
* scikit-learn
* jupyter

## Project Description

Human body activities can be classified based on data of joints or limbs. For example, firstly choose some critical joints which can generally and specifically define an activity (e.g. wrist, elbow, ankle, knee). Then being aware of the spacial coordinates of these joints can give a general picture of what the current activity is, which is also the basis for classification.

We deploy the software and equippment from Captury Live, whose GUI directly output spacial coordinates of all pre-defined joints, to create raw dataset for training our machine learning model to classify.

We preprocessed raw data from .csv file from Captury Live, including:

* visualize coordinates
* select joints
* calculate kinematic features
* windowlize frames (for classification of dynamic activities)
* calculate statistic featues

After the operation of preprocessing mentioned above, we split dataset into **trainset** and **testset** for model. Note that **testset** can also be provided with extra dataset instead of extracting from original dataset, which will be covered later.


## Getting Started

### 1) Files Structure

The whole directory structure is shown as follows. 
```
└── root/
    ├── train_dynamic_ML.py
    ├── train_static_ML.py
    ├── plot_features.ipynb
    ├── README.md
    ├── Archiv/
    ├── dataloader/
    |   └── ML_dataloader.py
    ├── datasets/
    |   ├── data_visualization.py
    |   ├── dataset_generation.py
    |   ├── desired_features_trial.yaml
    |   ├── desired_features.yaml
    |   └── raw_data_1/
    |       ├── split_method.yaml
    |       ├── unknown.NoHead.csv
    |       ├── x_data_UpperLowerBody.npy
    |       └── y_data_UpperLowerBody.npy
    ├── model/
    |   ├── dynamic_models.py
    |   └── static_models.py
    ├── result/
    |   ├── model_prediction_probability.png
    |   └── static_models.py
    └── util/
        ├── features.py
        ├── plots_dynamic.py
        └── utils.py
```

### 2) Generate dataset

The dataset can be customized based on raw data from .csv from Captury Live. Feel free to research on which **features** to select for dataset, which is directly used to train model.

#### Step1: Edit .csv

**Important!!!**

Due to version issue of Captury Live software on Ubuntu18, the .csv output has problems to number of columns in head-part, i.e. number of columns of these items differ from each other:

* 1st row: recording information
* 2nd row: number of frames
* 3rd row: names of coordinates / joints
* 4th row: coordinates signs
* 5th row: unit of coordinates
* last rows: camera calibration configuration

As a reusult, it is difficult to directly extract data from raw .csv with Pandas, which needs manual editing. Please follow the steps below:

* 1. manually delete first 5 rows and last rows of camera calibration configuration
* 2. rename .csv as ```unknown.NoHead.csv```
* 3. move ```unknown.NoHead.csv``` under path ```/dataset/name_of_your_raw_data/```

#### Step2: Check data by plotting

Run this script to plot the segment with default arguments:

```
cd dataset/
python data_visualization.py --function "check_ori_data" --single_data_path "dataset/name_of_your_raw_data" --start_frame 0 --end_frame 200
```

Also, you can edit the arguments which you want to check:

| args | type | Description |
|--|--|--|
| --single_data_path | str | edit to the path where your ```unknown.NoHead.csv``` is located |
| --start_frame | int | from which frame to plot |
| --end_frame | int | to which frame to plot |

#### Step3: Create split_method.yaml

#### Step4: Edit desired_features.yaml

#### Step5: Generate dataset

### 1. Label method

1. Static Activity

| label | Abbreviation | Description |
|-------|--------------|-------------|
|1| HandNear | hold hands near body |
|2| HandAway | hold hands away from body |
|3| HandOverHead | hands over head |
|4| PhoneLH | hold phone with left hand |
|5| PhoneRH | hold phone with right hand |
    
2. Dynamic Activity

| label | Abbreviation | Description |
|-------|--------------|-------------|
|1| Walking | Walking |
|2| Jumping | Jumping |
|3| Squating | Squating |
|4| Waving | Waving |
|5| Reaching | Reaching out for a cup |
|6| Drinking | Drinking from a cup |
|7| Boxing | Reaching out left and right arms alternatively |
|8| Bicep | Bicep curl |
|9| JumpJack | Jumping jack |

3. Attention on split method

e.g.

sequence    ##############################################

class       |-----walk-----|-----jump-----|-----wave-----|

outliers                   A              B

Original features, dist_ratio and angle_ratio might contain extreme large value (outliers):

dist_ratio = (dist[latter_framce,:] - dist[former_frame,:]) / frame_ratio
angle_ratio = (angle[latter_framce,:] - angle[former_frame,:]) / frame_ratio

Outliers usually appear during the transitional instance between 2 classes,
e.g. point A between walk and jump, point B between jump and wave

If calculate dist_ratio and angle_ratio following the formular above, outliers
will appear during the transitional instance between 2 classes, e.g.:

point A between walk and jump

point B between jump and wave

For now the solution is: calculate all features based on original data without
cutting by split method. And when define split_method.yaml, **ensure the frames
around transitional points are not included**.

e.g. walk is in frame 0-100, jump is in 100-200, wave is 200-300, (when original
data is sequential), then the split_method.yaml should be like:

Walk1:
  start: 0
  end: 98
  label: 1
Jump1:
  start: 102
  end: 198
  label: 2
Wave1:
  start: 202
  end: 300
  label: 3

In this way, the outliers would not enter final dataset for training.

This can be improved or re-written later.

### 3) Train model

#### Members:

|Name |
|--|
|Jialei Li |
|Apostolos Vrontos |

## Contact
* If you need more details of codes, please contact Email of [Jialei Li](mailto:jia.lei.li@rwth-aachen.de);
* If you need to know background or further application development of this project, please contact Email of [Apostolos Vrontos](mailto:a.vrontos@iaw.rwth-aachen.de)
