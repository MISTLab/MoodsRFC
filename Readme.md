## Expressive motion classification
This repository contains the python files to build, train and predict the 27 datasets of https://ieee-dataport.org/documents/expressive-motion-dancers.

# Python script
The main file is "classifyRF.py" which will build the datasets, train and classify each performance for each participants.
The "load_data.py" file is used to load and build the datasets and performances.
The "build_features_vector.py" build the vector of features that contained within each examples.
The "extract_features.py" contain the definition of the various features utilized in the final classifier of the article.

# Citation
If you use the code or the dataset please cite this work with:

St-Onge, D., Côté-Allard, U., Glette, K., Gosselin, B. and Beltrame, G. "Engaging Swarms: Expressive motion from Moods". [submitted to] Transaction on Human-Robot Interactions. 2018.