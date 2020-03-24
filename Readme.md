# Expressive motion classification
This repository contains the python files to build, train and predict the 27 datasets of https://ieee-dataport.org/documents/expressive-motion-dancers.

## Requirements
To run this code on Windows:
1. Install Python 3.6 via Anaconda (https://www.anaconda.com/download/)
2. With the anaconda prompt, install the dependent libraries for Python (scikit-learn, numpy and scipy):

``conda install scikit-learn``

## Python script
1. The main file is "classifyRF.py" which will build the datasets, train and classify each performance for each participants.
2. The "load_data.py" file is used to load and build the datasets and performances.
3. The "build_features_vector.py" build the vector of features that contained each examples.
4. The "extract_features.py" contain the definition of the various features used for the final classifier of the article below.

## Citation
If you use the code or the dataset please cite this work with:

> St-Onge, David, Côté-Allard, Ulysse, Glette, Kyrre, Gosselin, Benoit et Beltrame, Giovanni. 2019. « Engaging with robotic swarms: commands from expressive motion ». ACM Transactions on Human-Robot Interaction (THRI), vol. 8, nº 2.
