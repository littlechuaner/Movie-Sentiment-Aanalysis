#!/usr/bin/env python
# coding: utf-8

# In[42]:


import warnings
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

import xgboost as xgb

warnings.filterwarnings("ignore")


# In[2]:


# # Read data
train = pd.read_csv('./labeledTrainData.tsv', delimiter='\t', quoting=3)
test = pd.read_csv('./testData.tsv', delimiter='\t', quoting=3)
# get the number of training and test examples
n_train = len(train)
n_test = len(test)


# In[3]:


# # Data Cleaning and Processing
def review2words(review):
    """ function to convert input review into string of words """
    # Remove HTML
    review_text = BeautifulSoup(review, 'lxml').get_text() 

    # Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 

    # Convert to lower case, split into individual words
    words = letters_only.lower().split()                             

    # Join the words and return the result.
    return " ".join(words)


# get train label
train_y = train['sentiment'].values

# transform reviews into words list
train_review = list(map(review2words, train['review']))
test_review = list(map(review2words, test['review']))

# combine train and test reviews
all_review = train_review + test_review

# perform TF-IDF transformation
vectorizer = TfidfVectorizer(min_df=3, analyzer="word", strip_accents='unicode', 
                             sublinear_tf=True, stop_words='english', 
                             max_features=10000, ngram_range=(1, 2)) 

# fit and transform the data
all_features = vectorizer.fit_transform(all_review)

# trainsform into array
train_features = all_features[:n_train, :].toarray()
test_features = all_features[n_train:, :].toarray()


# In[45]:


# # RF
param_grid = {"max_depth": [10,20,30,50,80]}
# Creating the classifier
classifier = RandomForestClassifier(max_features='auto', max_depth=5 ,n_estimators=500, 
                                    random_state=2019, criterion='entropy', n_jobs=-1, verbose=1 )
grid_search = GridSearchCV(classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(train_features, train_y)
print(grid_search.best_score_)
print(grid_search.best_params_)


# In[46]:


# Initialize a Random Forest classifier
forest = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=2019, 
                                oob_score=True, max_features='auto',
                               max_depth=80) 

# Fit the forest to the training set
forest = forest.fit(train_features, train_y)

# make predictions
test_pred_rf = forest.predict(test_features)


# In[ ]:


# # Gradient Boosting 
parameters = {'loss': ['deviance', 'exponential'], 
              'learning_rate': [0.05,0.1,0.2,0.5],
             'n_estimators':[500],
             'max_depth':[10,20,50,80]}
classifier=GradientBoostingClassifier()
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(train_features, train_y)
best_accuracy = grid_search.best_score_
print("The best accuracy using gridSearch is", best_accuracy)
best_parameters = grid_search.best_params_
print("The best parameters for using this model is", best_parameters)


# In[28]:


# build the Gradient Boosting classifier
gbm = GradientBoostingClassifier(learning_rate=0.2, n_estimators=500,
                                 max_features='auto', max_depth=50,
                                 loss='exponential')
gbm = gbm.fit(train_features, train_y)

# make predictions
test_pred_gb = gbm.predict(test_features)


# In[ ]:


# # AdaBoost
parameters = { 'learning_rate': [0.05,0.1,0.2,0.5],
              'n_estimators':[500]}
classifier=AdaBoostClassifier()
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(train_features, train_y)
best_accuracy = grid_search.best_score_
print("The best accuracy using gridSearch is", best_accuracy)

best_parameters = grid_search.best_params_
print("The best parameters for using this model is", best_parameters)


# In[7]:


# build the AdaBoost classifier
adaboost = AdaBoostClassifier(n_estimators=500, learning_rate=0.2, 
                              algorithm='SAMME.R', random_state=2019)
adaboost = adaboost.fit(train_features, train_y)

# make predictions
test_pred_adb = adaboost.predict(test_features)


# In[ ]:


# # XGBoost
parameters = {'max_depth':[10,20,50,80], 
              'learning_rate': [0.05,0.1,0.2,0.5],
              'n_estimators':[500]}
classifier=xgb.XGBClassifier(objective="binary:logistic")
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(train_features, train_y)
best_accuracy = grid_search.best_score_
print("The best accuracy using gridSearch is", best_accuracy)

best_parameters = grid_search.best_params_
print("The best parameters for using this model is", best_parameters)


# In[38]:


xgboost = xgb.XGBClassifier(objective="binary:logistic", random_state=2019,
                            learning_rate= 0.2, n_estimators=500,
                            max_depth=50)
xgboost = xgboost.fit(train_features, train_y)

# make predictions
test_pred_xgb = xgboost.predict(test_features)


# In[39]:


test["sentiment"] = test["id"].map(lambda x: 1 if int(x.strip('"').split("_")[1]) >= 5 else 0)
y_test = test["sentiment"]
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
#ROC
rf_fpr, rf_tpr, rf_thresold = roc_curve(y_test, test_pred_rf)
gb_fpr, gb_tpr, gb_threshold = roc_curve(y_test, test_pred_gb)
adb_fpr, adb_tpr, adb_threshold = roc_curve(y_test, test_pred_adb)
xgb_fpr, xgb_tpr, xgb_threshold = roc_curve(y_test, test_pred_xgb)


def graph_roc_curve_multiple(rf_fpr, rf_tpr, gb_fpr, gb_tpr, adb_fpr, adb_tpr, xgb_fpr, xgb_tpr):
    plt.figure(figsize=(16,8))
    plt.title('ROC Curve \n 4 Classifiers', fontsize=18)
    plt.plot(rf_fpr, rf_tpr, label='Random Forest AUC: {:.4f}'.format(roc_auc_score(y_test, test_pred_rf)))
    plt.plot(gb_fpr, gb_tpr, label='Gradient Boosting AUC: {:.4f}'.format(roc_auc_score(y_test, test_pred_gb)))
    plt.plot(adb_fpr, adb_tpr, label='Adaboost AUC: {:.4f}'.format(roc_auc_score(y_test, test_pred_adb)))
    plt.plot(xgb_fpr, xgb_tpr, label='XGBoost AUC: {:.4f}'.format(roc_auc_score(y_test, test_pred_xgb)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.legend()
    
graph_roc_curve_multiple(rf_fpr, rf_tpr, gb_fpr, gb_tpr, adb_fpr, adb_tpr, xgb_fpr, xgb_tpr)
plt.savefig("ROC2.png",dpi=800)
plt.show()


# In[40]:


from sklearn.metrics import confusion_matrix
rf_cf = confusion_matrix(y_test, test_pred_rf)
gb_cf = confusion_matrix(y_test, test_pred_gb)
adb_cf = confusion_matrix(y_test, test_pred_adb)
xgb_cf = confusion_matrix(y_test, test_pred_xgb)

fig, ax = plt.subplots(2, 2,figsize=(22,12))


sns.heatmap(rf_cf, ax=ax[0][0], annot=True, cmap=plt.cm.copper)
ax[0, 0].set_title("Random Forests \n Confusion Matrix", fontsize=14)
ax[0, 0].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[0, 0].set_yticklabels(['', ''], fontsize=14, rotation=360)

sns.heatmap(gb_cf, ax=ax[0][1], annot=True, cmap=plt.cm.copper)
ax[0][1].set_title("Gradient Boosting \n Confusion Matrix", fontsize=14)
ax[0][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[0][1].set_yticklabels(['', ''], fontsize=14, rotation=360)

sns.heatmap(adb_cf, ax=ax[1][0], annot=True, cmap=plt.cm.copper)
ax[1][0].set_title("AdaBoost \n Confusion Matrix", fontsize=14)
ax[1][0].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[1][0].set_yticklabels(['', ''], fontsize=14, rotation=360)

sns.heatmap(xgb_cf, ax=ax[1][1], annot=True, cmap=plt.cm.copper)
ax[1][1].set_title("XGBoost \n Confusion Matrix", fontsize=14)
ax[1][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[1][1].set_yticklabels(['', ''], fontsize=14, rotation=360)

plt.savefig("cm.png",dpi=800)
plt.show()

