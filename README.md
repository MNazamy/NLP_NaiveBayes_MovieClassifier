# Naive-Bayes Classifier 

Author: Matin Nazamy

## Features Used:
- Bag-Of-Words Feature Vector of size 89527

## Techniques Used:
- Add-One-Smoothing

## How To Run This Program

1. Ensure you have the necessary data files in the same root directory

- *NB.py*
- *preprocess.py*
- *small-dataset/*
- *movie-review-HW2/*
    - **this is also provided, just unzip the 'aclIMDB.zip' file into a directory called 'movie-review-HW2'. And that directory should be in the same level as the python files**


2. Run NB.py. The outputs will be found in:

- *./small-output/*
- *./movie-review-output/*


3. If you run NB.py while the movie-review-output files exist, it will not do preprocessing again. It will jump straight into training and testing. If you wish to do preprocessing again, please delete the movie-review-output directory.


## How to Read This Program

1. Go to the specific output file you are looking for (either "small-output/" or "movie-review-output/"
2. Open the file *output.txt*
3. At the bottom of this file will be the Accuracy and other scoring metrics.
    - These scoring metrics represent how well the model did when tested against the test data, and being trained on the train data.
