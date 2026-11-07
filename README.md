## Problem Statement  
### Mobile Price Prediction
A mobile manufacturing company wants to categorize mobile phones into four price ranges based on product specifications such as `battery power`, `RAM`, `display resolution`, `connectivity features`, etc.

Using the provided dataset, the objective is to:  
1. Prepare the data for multi-class classification  
2. Train and compare multiple classifiers  
3. Identify the best model to predict price range  
4. Report inferences from results and evaluation metrics  
5. Deploy using Steamlit.  

## Dataset Description
The dataset consists of multiple numerical and categorical features describing the specifications of mobile phones, with `price_range` serving as the target variable. The objective is to train multiple supervised machine learning classification models that can assign a mobile phone to the appropriate price category based on its specifications and identify the best one. The problem is formulated as multiclass classification, where the model learns the relationship between device characteristics and the corresponding price segment.  


## GitHub Repository Link  
[Repository Link](https://github.com/anikets349/ML-Assignment.git)


## Models used  
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| ------------- | ------ | ------ | ------ | ------ | ------ | ------ |
| `Logistic Regression` | 0.947 | 0.996 | 0.947 | 0.947 | 0.947 | 0.930 |
| `Decision Tree` | 0.829 | 0.886 | 0.831 | 0.829 | 0.829 | 0.773 |
| `KNN` | 0.550 | 0.779 | 0.564 | 0.550 | 0.552 | 0.402 |
| `Gaussian NB` | 0.798 | 0.945 | 0.802 | 0.798 | 0.799 | 0.731 |
| `Multinomial NB` | 0.538 | 0.777 | 0.574 | 0.538 | 0.543 | 0.392 |
| `SVM` | 0.857 | 0.952 | 0.861 | 0.857 | 0.858 | 0.810 |  

## Observations
| ML Model Name | Observation about model performance |
| -------------- | --------------------------------------------------------|
| `Logistic Regression` | Performs best overall, with 94.7% test accuracy and F1, along with the highest AUC (0.996) and MCC (0.93). |
| `Decision Tree` | Shows overfitting, with 100% training accuracy dropping to 82.9% test accuracy. |
| `KNN` | Performs poorly, with test accuracy of only 55.0%. |
| `Gaussian NB` | Provides moderate performance, with approximately 79.8% test accuracy. |
| `Multinomial NB` | Performs poorly, with test accuracy of only 53.8%. |
| `SVM` | Second-best model, achieving 85.7% test accuracy and F1. |  

Overall best model for dataset: `Logistic Regression`