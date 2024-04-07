# DLNLP_assignment_24

## Description

- Kaggle Datasets website: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data
- Twitter Sentiment Analysis
- 3-class problem

## Prepare

- you need download the competiton dataset from https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data
- Please download all datasets, include "twitter_training.csv" and "twitter_validation.csv" !!!!!
- Put the two csv into Datasets folder !!!!!
- you need download the "glove.6B.300d.txt" from https://www.kaggle.com/datasets/aellatif/glove6b300dtxt !!!!!
- Put the "glove.6B.300d.txt" with main.py in the same location. !!!!! 

## Documents

- main.py (run the project)
- README (Introduce the project)
- plot (The folder uses to store the result diagram from the main.py)
- Datasets (The folder uses to store the image data)
- A (The folder uses to store the whole code about LSTM model)
- B (The folder uses to store the whole code about Bi-LSTM model)
- C (The folder uses to store the whole code about BERT model)
- environment.yaml (The copy of student's environment)

## Point

- You can find lots of jupyter notebooks in the folder A, B and C.
- These jupyter notebooks just like the history, which include all the results made by student.
- If you want to find the results of model with different parameters, you can see these jupyter notebooks.
- OR you can change the parameters in .py file directly.
- In main.py, you can see three methods. If you want to run one of these methods, you can remove comments and comment the code of other methods.

## Packet required
To run the code in this project, the following packages are required:
- `scikit-learn`
- `transformers`
- `matplotlib`
- `tensorflow`
- `tqdm`
- `seaborn`
- `torch`
- `pandas`
- `numpy`
OR
The required environment has been export to file "environment.yaml". Use the following conda instruction to finish the environment setting.