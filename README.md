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