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

### #TODO (how to get raw data)
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is being kept [here](Repo folder containing raw data) within this repo.

    
3. Data processing/transformation scripts are being kept [here](Repo folder containing data processing scripts/notebooks)
4. etc...

*If your project is well underway and setup is fairly complicated (ie. requires installation of many packages) create another "setup.md" file and link to it here*  

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



#### Members:

| Name |
|--|
| Jialei Li |
| Apostolos Vrontos |

## Contact
* If you need more details of codes, please contact [Jialei Li](mailto:jia.lei.li@rwth-aachen.de);
* If you need to know background or further application development of this project, please contact [Apostolos Vrontos](mailto:a.vrontos@iaw.rwth-aachen.de)
