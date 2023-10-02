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

### 1) Configure env

Firstly create a new env and install all necessary dependencies. If you are using **Anaconda** on **Linux**, you can directly run the following commands in terminal:

```
conda create --name ActClass python
pip install numpy
pip install pandas
pip install -U matplotlib
pip install -U scikit-learn
pip install pyyaml
```

### 2) Record shot with Captury Live

The original data of this project is from Captury Live, which outputs abandunt forms of data. We only use the ```unknown.csv```.

Please build the original data according to the following points in order to run this repo successfully:

1. Save each ```unknown.csv``` in a sub-folder, without change csv's name
2. Name the sub-folders with style: **TrainOrTest_ExpGroup_2DigitActLabel_ActAbbreviation**, e.g. **Train_Dynamic_Jialei_00_None**.
  Attention:
    1) ExpGroup consists of DynamicOrStatic_UserName
    2) Make sure the ActLabel is 2-digital starting with 0
    3) use the same ActAbbreviation when editing ```split_method.yaml``` for testset in **3) Step4**
3. Save train data in ```ori_csv/ExpGroup/trainset``` and test data in ```ori_data/ExpGroup/testset```. For example, a folder for one ExpGroup **Static_Jialei** has such structure:

```
└── ori_csv/
    └── Static_Jialei/
        ├── testset/
        |   └── Test_Static_Jialei/
        |       └── unknown.csv
        └── trainset/
            ├── Train_Static_Jialei_00_None/
            |   └── unknown.csv
            ├── Train_Static_Jialei_01_ExtendArm/
            |   └── unknown.csv
            ├── Train_Static_Jialei_02_RetractArm/
            |   └── unknown.csv
            ├── Train_Static_Jialei_03_HandOverHead/
            |   └── unknown.csv
            └── Train_Static_Jialei_04_Phone/
                └── unknown.csv
```

### 3) Create dataset

Follow the steps to prepare and examine data for creating dataset.

#### Step1: Generate exp_group & cut csv

Firstly, **exp_group** is the folder containing trainset and testset data of one single user, whose name follow the naming-style **Dynamic/Static_username**. Folder structure of a complete **exp_group** is like the following example:

```
└── datasets/
    └── Dynamic_user1/
        ├── testset/
        |   └── Test_Dynamic_user1/
        |       ├── split_method.yaml
        |       └── unknown.NoHead.csv
        └── trainset/
            ├── Train_Dynamic_user1_00_act1/
            |   ├── split_method.yaml
            |   └── unknown.NoHead.csv
            ├── Train_Dynamic_user1_01_act2/
            |   ├── split_method.yaml
            |   └── unknown.NoHead.csv
            └── Train_Dynamic_user1_02_act3/
                ├── split_method.yaml
                └── unknown.NoHead.csv
```

The above exp_group **Dynamic_user1** defines 3 classes of activities **act1**, **act2** and **act3**. Due to recording methods, shot of each class of activity in **trainset** are saved separately, while **testset** not. Please refer to report for details.

Note that the ```/config/class_ExpName.yaml``` should follow naming style: **actName: actLabel**, actName is the abbreviation of activity name, which will be presented in plot of performance at last, actLabel is integer representing class of activity. Refer to example ```/config/class_Static.yaml```.

Secondly, original ```unknown.csv``` files from Captury Live has redandunt information. Hence, some of lines need to be cut off.

In order to generate a exp_group under ```/dataset```:

1. Create a new ```/config/class_ExpName.yaml``` according to example ```/config/class_Static.yaml```
2. Edit arguments in ```prepare_exp_group.py``` according to instructions inside
3. Run with: ```python prepare_exp_group.py```

#### Step2: Check data by plotting

Check the Original Data $Arr_{ori}$ from ```unknown.NoHead.csv``` from **Step1** with function **check_ori_data**, by editing the arguments according to instruction in ```data_visualization.py``` and running it with:

```
python data_visualization.py
```

#### Step3: Edit ```split_methods.yaml``` for testset

Note that due to recording methods, **trainset** consists of shots with same frames, each of shot contains only one activity, hence the ```split_methods.yaml``` can be generated automatically by ```generate_split_methods_trainset.py``` in **step1**. While **testset** consists of only one shot with all activities for testing, the ```split_methods.yaml``` need to be manully edited with help of function **check_ori_data** in ```data_visualization.py```.

Refer to **Step2** for how to use ```data_visualization.py```.

Refer to example below to manully edit ```split_methods.yaml``` for testset(number after activity name indicates how many same activities appear in current shot, e.g. None1, None2):

```
None1:            # None class is for unexplicit or transitional movement between defined classes
  start: 0        # starts from the "end" of previous class
  end: 220        # ends at the "start" of next class
  label: 0        # label as 0
Boxing1:          # Name = Abbreviation of activity + x-th segment with same activity
  start: 220      # start: from which frame belong to this segment of activity
  end: 480        # end: to which frame belong to this segment of activity
  label: 7        # label: customized label of this activity
None2:
  start: 480
  end: 680
  label: 0
Boxing2:
  start: 680
  end: 860
  label: 7
```

#### Step4: Edit desired_features.yaml

Edit ```config/desired_features.yaml``` to determine which features you want to add into your final dataset to train. E.g.:

```
desired_dists:
  - LHandEnd_head
  - LWrist_head
  - LElbow_head
desired_angles:
  - LHandEnd_LWrist_LElbow
  - LWrist_LElbow_LShoulder
```

The rule of defining feature's name is as following:

* distance features: joint1_joint2
* angle features: joint1_joint2_joint3 (joint2 is vertex of the angle)

All joints' names to be selected and corresponding index is shown below:

| Index | Abbreviation | Joint |
|--|--|--|
|0| LWrist | left wrist |
|1| LElbow | left elbow |
|2| LShoulder | left shoulder |
|3| RWrist | right wrist |
|4| RElbow | right elbow |
|5| RShoulder | right shoulder |
|6| LToe | left toe |
|7| LAnkle | left ankle |
|8| LKnee | left knee |
|9| LHip | left hip |
|10| RToe | right toe |
|11| RAnkle | right ankle |
|12| RKnee | right knee |
|13| RHip | right hip |
|14| LClavicle | left clavicle |
|15| LHandEnd | end of left hand |
|16| LToesEnd | end of left toe |
|17| RClavicle | right clavicle |
|18| RHandEnd | end of right hand |
|19| RToesEnd | end of right toe |
|20| spine1 | spinal joint between left hip and right hip |
|21| spine2 | 2nd spinal joint from spine1 |
|22| spine3 | 3rd spinal joint from spine1 |
|23| spine4 | 4th spinal joint from spine1 |
|24| spine5 | 5th spinal joint from spine1 |
|25| head | head vertex |

<img src="Archive/joint_index.jpg" alt="My Image" width="150" height="450">

Refer to report for more details.

#### Step5: Verify features

Use function **verify_before_output** in ```data_visualization.py``` to check whether there are errors in calcualtion of Frame Feature Array $Arr_{ff}$. Refer to **Step2** for how to use ```data_visualization.py```.

### 3) Plot meta-feature

To get a intuition of periodical behavior of window Window Feature Array $Arr_{wf}$, run ```plot_features.ipynb``` to visualize meta-features. Refer to ```plot_features.ipynb``` for usage. Refer to report for details about **windowlization** and **calculation of meta-features**.

### 4) Train & Test

#### 1. Train model

Edit arguments in ```train.py``` according to instructions inside. And run by:

```
python train.py
```

Note that

1. **Cross Test** means that train on trainset of train_exp_group, i.e. from user1, and test on testset of test_exp_group, i.e. from user2.
1. **NonCross Test** means that train and test on same exp_group, i.e. current train_exp_group from one user

#### 2. Outputs

Every time ```train.py``` is successfully run, a folder with outputs is generated. In the folder, there are 4 forms of results:

1. PNG picture:         performance of accuracy of testing, both classification result and truth
2. arg.yaml:            configuration of current experiment, for extra testing in ```test.py``` and loading later, if necessary
3. miscls_index.txt:    index of misclassified windows, indices of frames making up these windows
4. model.pickle:        trained model in current experiment, for extra testing in ```test.py``` and loading later, if necessary

#### 3. Extra testing

You can define which trainset and which testset to train and test, but you can also load a trained model output from ```train.py``` and designate a new testset to test it with ```test.py```. Refer to instructions inside ```test.py``` for usage.

### 5) Post processing

Get frames of misclassified windows in ```miscls.txt```, and visualize corresponding windows ```data_visualization.py```, under function **post_process**. Refer to **3) Step2** for usage.

## Contact

The members in this project:

|Name |
|--|
|Jialei Li |
|Apostolos Vrontos |

* If you need more details of codes, please contact Email of [Jialei Li](mailto:jia.lei.li@rwth-aachen.de);
* If you need to know background or further application development of this project, please contact Email of [Apostolos Vrontos](mailto:a.vrontos@iaw.rwth-aachen.de)
