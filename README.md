# Project Description
Implementation of a Economical Crisis forecasting model using [Bluwstein & Others](https://editorialexpress.com/cgi-bin/conference/download.cgi?db_name=EEAESEM2019&paper_id=1163) as reference. Used a reduced cluster of countries in a shorter range of years in order to achieve a better model.

# Dataset
JST-Macrohistory DataBase. Almost used in every Crisis Forecasting study. Find it in the next document:  
http://www.macrohistory.net/JST/JSTdocumentationR2.pdf

# Files
## src/train.py
This program executes the feature engineering and model hyperparametrization processes. The classifiers used in this implementation are:
- Logistic Regression
- Random Forest
- Extremely Randomized Trees
- Extreme Gradient Boosting
- Multilayer Perceptron  
The method Grid Search is used in order to find the best hyperparameters.  
After running the classifiers, the program generates a file named report.txt. This report contains the best hyperparameters for each classififer, the ROC-AUC score obtained with the best parameters and the total time of execution.
## src/utils.py

## src/config.py

## data/JSTdatasetR4.xlsx

## run.bat

