# Importing required libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


def select_data_diagnosis(data, diagnosis="healthy"):
    """According to the diagnosis, outputs data for splitting and prediction
       
       Input: dataset, string of type of diagnosis used for the prediction
              where healthy is the default. """
    
    global diagnosis_string
    
    # check if na values
    if data.isnull().values.any() == True:
        print("Watch out, there are some NA values")
        
    # check if diagnosis is healthy or benign
    if diagnosis == "healthy":
        # drop benign data and return data
        used_data = data.loc[data["diagnosis"] != 2]  
        diagnosis_string = "healthy"
        return used_data
        return diagnosis_string
    elif diagnosis == "benign":
        # drop healthy data and return data
        used_data = data.loc[data["diagnosis"] != 1] 
        diagnosis_string = "benign"
        return used_data
        return diagnosis_string
    else:
        print("Please check the diagnosis input, has to be 'healthy' or 'benign'.")



def predict_roc(x_test, y_test, model, model_string):
    """Function which predicts the ROC-curve with corresponding AUC.
   
       Input: x test data, y test data, the model and the string of the model.
       
       Works with every classification model. """
    
    global fpr
    global tpr
    global thresholds
    global model_auc
    
    ## Predict the model
    # Predict with probabilities and keep probabilities for the negative outcome only
    pred_prob = model.predict_proba(x_test)[:, 1]
           
    # calculate scores
    model_auc = roc_auc_score(y_test, pred_prob)
    
    # summarize scores
    print(f"{model_string}: ROC AUC=%.3f" % (model_auc))
    
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(y_test, pred_prob, pos_label=3)
    
    

    
### Prediction of the dataset
# Load the dataset
data = pd.read_csv("data/pancreatic_data.csv")

## Data selection for predicting PDAC with urine biomarkers
# Drop all irrelevant variables(sample_id, patient_cohort, sample_origin, plasma_ca19_9, REG1A)
data_select = data[["age", "diagnosis", "creatinine", "sex",
                    "LYVE1", "REG1B", "TFF1"]]

# change male to 0 and female to 1
data_select.sex[data_select.sex == "M"] = 0
data_select.sex[data_select.sex == "F"] = 1

# select healthy and benign data, uncomment the one you want to use in the prediction
used_data = select_data_diagnosis(data_select)
# used_data = select_data_diagnosis(data_select, "benign")

# Split data in predictor and target variables
y = used_data.diagnosis
x = used_data.drop("diagnosis", axis=1)
x.shape, y.shape # sanity check

# Normalize x data
scale_x = StandardScaler()
x = scale_x.fit_transform(x)

# Split the data in train and test data (50/50)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=42, shuffle=True)
x_train.shape, y_train.shape # sanity check
x_test.shape, y_test.shape # sanity check

# Prepare 10-fold LOOCV (Leave One Out Cross Validation)
cv = KFold(n_splits=10, random_state=42, shuffle=True)


# ## Random forest model
# Creating and fitting the model
model_rf = RandomForestClassifier(n_estimators=1500, random_state=42, 
                                  max_depth=50, max_features='log2',
                                  min_samples_leaf=10, min_samples_split=10) 
model_rf.fit(x_train, y_train)

# perform and calculate CV
scores_rf = cross_val_score(
    model_rf, x_train, y_train, scoring="roc_auc", cv=cv)
print("Random forest ROC AUC CV scores: %.3f" % (np.mean(scores_rf)))

## Predicting the model with ROC curve function
predict_roc(x_test, y_test, model_rf, "Random Forest")

# plot rf
plt.plot(fpr, tpr, marker=".", 
          label=f"Random Forest (AUC = {round(model_auc, 3)})")

print("-"*80)


## Logistic regression model
C = 0.1

# Creating and fitting the model
model_lr = LogisticRegression(solver="lbfgs", penalty="l2", 
                              random_state=42, C=C)
model_lr.fit(x_train, y_train)

# perform and calculate CV
scores_lr = cross_val_score(
    model_lr, x_train, y_train, scoring="roc_auc", cv=cv)
print("Logistic regression ROC AUC CV scores: %.3f" % (np.mean(scores_lr)))

## Predicting the model with ROC curve function
predict_roc(x_test, y_test, model_lr, "Logistic regression")

# plot lr
plt.plot(fpr, tpr, marker=".", 
          label=f"Logistic regression (AUC = {round(model_auc, 3)})")

print("-"*80)


## SVM linear model
C = 41.8

# Creating and fitting the model
# Linear kernel
model_svm = SVC(kernel="linear", gamma=0.097, C=C, probability=True) 
model_svm.fit(x_train, y_train)

# perform and calculate CV
scores_svm = cross_val_score(
    model_svm, x_train, y_train, scoring="roc_auc", cv=cv)
print("SVM ROC AUC CV scores: %.3f" % (np.mean(scores_svm)))

## Predict and plot ROC curve with function
predict_roc(x_test, y_test, model_svm, "SVM")

# Plot all the roc curves in one graph
plt.plot(fpr, tpr, marker=".", 
          label=f"SVM (AUC = {round(model_auc, 3)})")
plt.plot([0,1],[0,1], linestyle='--', color='#737373')
plt.legend()
plt.xlabel("1-specificity")
plt.ylabel("Sensitivity")
plt.title(f"ROC curves {diagnosis_string} vs PDAC")
plt.savefig(f"data/ROC curves {diagnosis_string}.png")
plt.show()

##########################################################################################

# ## Tuning hyperparameters random forest, delete # to perform search
# # Create dictionary with some parameters
# model = RandomForestClassifier(random_state=42)
# n_estimators=[500, 800, 1500, 2500, 5000]
# max_features=["auto", "sqrt", "log2"]
# max_depth = [10, 20, 30, 40, 50]
# max_depth.append(None)
# min_samples_split = [2, 5, 10, 15, 20]
# min_samples_leaf = [1, 2, 5, 10, 15]

# # define grid search
# grid = dict(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth,
#             min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
# # Create gridsearch object, fit on training data
# grid = RandomizedSearchCV(model, param_distributions=grid, cv=cv, n_iter=500, 
#                           refit=True, verbose=2, n_jobs=1)
# grid.fit(x_train, y_train)
# # Print best parameters
# print(grid.best_estimator_)
# #max_depth=50, max_features='log2', min_samples_leaf=10,
# # min_samples_split=10, n_estimators=1500,


# ## Tuning hyperparameters Logistic regression, delete # to perform search

# # Create gridsearch object, fit on training data
# grid = RandomizedSearchCV(model, param_distributions=grid, refit=True, verbose=2)
# grid.fit(x_train, y_train)
# # Print best parameters
# print(grid.best_estimator_)
# # Best estimators: "lbfgs", l2, c=0.1


# ## Tuning hyperparameters for SVM, delete # to perform search
# # Create dictionary with some parameters
# #param_grid = {
# #   "C": [0.1, 1, 10, 100], "gamma": [1,0.1,0.01,0.001],
# #9    "kernel": ["rbf", "poly", "linear"]}
# param_grid = {"C": expon(scale=100), "gamma": expon(scale=.1),
#   "kernel": ["rbf", "poly", "linear"], "class_weight":["balanced", None]}

# # Create gridsearch object, fit on training data
# grid = RandomizedSearchCV(SVC(), param_grid, refit=True, verbose=2)
# grid.fit(x_train, y_train)

# # Print best parameters
# print(grid.best_estimator_)