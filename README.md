# Intro

Our codes are based on https://github.com/JacobYuan7/DIN-Group-Activity-Recognition-Benchmark.  
I deeply appreciate their efforts.

## Environment
python 3.10.2

And you can use requirements.txt
```
pip install -r requirements.txt
```

# Data preparation
## 1. Download dataset
You can download daatset from the following url.  
These dataset are required to place in data/ in the repository as follows:

* Volleyball dataset (data/volleyball/videos)  
https://github.com/mostafa-saad/deep-activity-rec

* Collective Activity dataset (data/collective)  
https://cvgl.stanford.edu/projects/collective/collectiveActivity.html


## 2. Training
* You can change parameters of the model by editing the files located in scripts (e.g., scripts/train_volleyball_stage2_gr.py).
* Trained model are also published in here (https://drive.google.com/drive/folders/1UnwII6cHG-5SMVPAHwweO92TUwXQfKqt?usp=drive_link).
* trained models required to place in result/ (e.g., result/GAFL_PAC_VOL).

### 2.1 Volleyball dataset

* Ours
```
python scripts/train_volleyball_stage2_gr.py
```
The following folder contains the trained models.
1. GAFL_PAC_VOL (GAFL-PAC)
2. GAFL_PAF_VOL (GAFL-PAF)

### 2.2 Collective Activity dataset

* Ours
```
python scripts/train_collective_stage2_gr.py
```
The following folder contains the trained models.
1. GAFL_PAC_CAD (GAFL-PAC)
2. GAFL_PAF_CAD (GAFL-PAF)

## 3. Evaluation
### 3.1 Volleyball dataset
You can choose the model that you would like to evaluate in the bash file script.

* Ours
```
bash ./evaluation_vol.bash
```

### 3.2 Collective Activity dataset

* Ours
```
bash ./evaluation_cad.bash
```
