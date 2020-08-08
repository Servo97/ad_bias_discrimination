# Bias Detection and Mitigation in Advertisements

Inspired by: 
https://devpost.com/software/impact-of-user-personality-for-advertisement-recommendation

Data has been preprocessed into train and test splits. 
Original data in the wild can be found at: https://www.kaggle.com/groffo/ads16-dataset

* Deep_learning-Advertisement1 -> multi label classification
* DLA-Binary                   -> binary classification
* DLA-SMOTE                    -> Reducing class imbalance using SMOTE
* DLA-SMOTE_\<category>        -> Impact of data augmentation on each category. (Reduction of bias.)