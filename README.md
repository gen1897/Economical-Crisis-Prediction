# Project Description
Implementation of a Economical Crisis forecasting model following [Bluwstein & Others](https://editorialexpress.com/cgi-bin/conference/download.cgi?db_name=EEAESEM2019&paper_id=1163) as principal reference. Used a reduced cluster of countries in a shorter range of years in order to achieve a better model.  

Python 3 required. To install necessary libraries run:
- *pip install pandas numpy scikit-learn xgboost*

# Dataset
JST-Macrohistory DataBase. Used almost in every Crisis Forecasting paper. Find it in the next document:  
http://www.macrohistory.net/JST/JSTdocumentationR2.pdf

# Files

## src/train.py
Main program file.  
Executes the feature engineering and model hyperparametrization processes. The classifiers used in this implementation are:
- Logistic Regression
- Random Forest
- Extremely Randomized Trees
- Extreme Gradient Boosting
- Multilayer Perceptron  

The method Grid Search is used in order to find the best hyperparameters.  
After running the classifiers, the program generates a file named report.txt. This report contains the best hyperparameters for each classififer, the ROC-AUC score obtained with the best parameters and the total time of execution.

## src/utils.py
Contains useful functions used in *src/train.py*. Each function is explained through comments in the file.

## src/config.py
Contains static variables used in *src/train.py*. The behaviour of the models can be changed modifying this variables. Variables included are:
- DATASET_NAME: Dataset path.
- SHEET_NAME: Contains the excel's sheet where dataset is stored.
- COUNTRIES: Countries used in the model.
- POST_WAR: When *True*, the model only used data posterior to 1949, when is the best date to start after WWII.
- PARAM_GRID: Contains the parameters grid for each classifier.
- CLASSIFIERS: Classifiers used.

## data/JSTdatasetR4.xlsx
JST dataset as given by the authors. Excel format. Data con be found in *Data* sheet.

## run.bat
Batch that runs *src/train.py*.
