# Advanced_Data_Analytics_University_Project
Group project for the Advanced Data Analytics course (SoSe 2026): EDA and machine learning pipeline.

## Task
The goal is to predict whether a customer of a car insurer will file a claim in the following year. Basis is the Porto Seguro Safe Driver Prediction Dataset from OpenML (ID: 42742). The Dataset and task originate from a Kaggle Competition in 2017. 

## Metric
Since the target feature is heavily imbalanced we opted to using the ROC-AUC metric as the main evaluation metric in developing and optimizing machine learning models.

## Dataset
|||
|---|---|
|Source|OpenML (ID 42742)|
|Link|https://www.openml.org/search?type=data&status=active&id=42742&sort=runs|
|Target|target(0,1)|
|Rows|595,212|
|Positive target rate|~3.6%|
|Missing value encoding|nan|

Since the dataset is not stored in this repository the recommended fetch implementation is:
```python
from sklearn.datasets import fetch_openml
df = fetch_openml(data_id=42742, as_frame=True).frame
```

## Feature naming scheme
|Feature name element|Meaning|
|---|---|
|ps_ind_*|attributes of the policy holder (undisclosed)|
|ps_reg_*|regional attributes (undisclosed)|
|ps_car_*|attributes of the insured vehicle (undisclosed)|
|ps_calc_*|calculated attributes (undisclosed)|
|*_bin|feature dtype == binary|
|*_cat|feature dtype == categorical|

## Data split
To ensure the comparability of the members work, the dataset was split in an 80/10/10, train/validation/test split via indexation. The .npy files containing said split indices, can be found via the /data/processed/ path.

## Folder Structure
```
.
├─ data
|   ├─ processed            # split files + local folder for processed data
|   └─ raw                  # local folder for raw data
├─ group
|   ├─ models               # group folder for saving important models
|   └─ notebooks            # group folder for notebooks made collaboratively
├─ members
|   ├─ E_S                  # member folder for E_S work progress
|   ├─ J_D                  # member folder for J_D work progress
|   │   └─ notebooks
|   ├─ L_L                  # member folder for L_L work progress
|   │   └─ notebooks_LL
|   └─ S_S_R                # member folder for S_S_R work progress
|       └─ notebooks
├─ .gitignore
├─ README.md
├─ environment.yml
└─ requirements.txt
```

## .gitignore
In order to not artificially bloat this repo the .gitignore was made to keep .db files, saved models etc from being pushed, until the team will decide otherwise.


## Setup
### Conda
```bash
conda env create -f environment.yml
conda activate ADA
```
### venv
```bash
pip install -r requirements.txt
```

## Future steps
This Repo is to be expanded upon after being screened and graded. Work will stop on the 30th August 2026 and will possibly resume after the grading process.