#!/usr/bin/env python
# coding: utf-8

# ## Predicting heart disease using machine learning
# 1. Problem definition
# 2. Data
# 3. Evaluation
# 4. Features
# 5. Modelling
# 6. Experimentation

# ### 1. Problem Statement
# Given clininal parameters about a patient, predict whether or not they have a heart disease.

# ### 2. Data
# The original data came from the Cleveland data from the UCI Machine Learning Repository
# https://archive.ics.uci.edu/dataset/45/heart+disease
# <br>
# Kaggle version https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
# <br>

# ### 3. Evaluation
# If we can reach 95% accuracy at predicting whether or not a patient has heart disease during the proof of concept, we'll pursue the project.

# ### 4. Features
# **Create data dictionary**
# 1. age - age in years
# 2. sex - (1 = male; 0 = female)
# 3. cp - chest pain type
# * 0: Typical angina: chest pain related decrease blood supply to the heart
# * 1: Atypical angina: chest pain not related to heart
# * 2: Non-anginal pain: typically esophageal spasms (non heart related)
# * 3: Asymptomatic: chest pain not showing signs of disease
# 4. trestbps - resting blood pressure (in mm Hg on admission to the hospital) anything above 130-140 is typically cause for concern
# 5. chol - serum cholestoral in mg/dl
# 6. serum = LDL + HDL + .2 * triglycerides
# * above 200 is cause for concern
# * fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)'>126' mg/dL signals diabetes
# 7. restecg - resting electrocardiographic results
# * 0: Nothing to note
# * 1: ST-T Wave abnormality
# * -- can range from mild symptoms to severe problems
# * -- signals non-normal heart beat
# * 2: Possible or definite left ventricular hypertrophy
# * -- Enlarged heart's main pumping chamber
# 8. thalach - maximum heart rate achieved
# 9. exang - exercise induced angina (1 = yes; 0 = no)
# 10. oldpeak - ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more
# 11. slope - the slope of the peak exercise ST segment
# * 0: Upsloping: better heart rate with excercise (uncommon)
# * 1: Flatsloping: minimal change (typical healthy heart)
# * 2: Downslopins: signs of unhealthy heart
# 12. ca - number of major vessels (0-3) colored by flourosopy
# * colored vessel means the doctor can see the blood passing through
# * the more blood movement the better (no clots)
# 13. thal - thalium stress result
# * 1,3: normal
# * 6: fixed defect: used to be defect but ok now
# * 7: reversable defect: no proper blood movement when excercising
# 14. target - have disease or not (1=yes, 0=no) (= the predicted attribute)

# In[3]:


import sklearn
sklearn.__version__


# ##### Preparing the tools

# In[4]:


# Regular EDA (exploratory data analysis) and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

# Models from Scikit_Learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve


# ##### Load Data

# In[5]:


df = pd.read_csv("heart-disease (1).csv")
df


# In[6]:


df.shape


# ##### Data Exploration (EDA)
# 1. What queation(s) are you trying to solve?
# 2. What kind of data do we have and how do we treat different types?
# 3. What's missing from the data and how do you deal with it?
# 4. Where are the outliers and why should you care about them?
# 5. How can you add, change or remove features to get more out of your data?

# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


# find out how many of items are in each class
df["target"].value_counts()


# In[11]:


df["target"].value_counts().plot(kind='bar', color=['salmon', 'lightblue']);


# In[12]:


df.info()


# In[13]:


# any missing values?
df.isna().sum()


# In[14]:


df.describe()


# ##### Heart disease frequency according to sex

# In[15]:


df.sex.value_counts()
# Male = 1
# Female = 0


# In[16]:


# Compare target column with sex column
pd.crosstab(df.target, df.sex)


# In[21]:


# we can infer
# (72/96) * 100 = 75% women in our dataset have a chance of heart disease
# (93/207) * 100 = 44.92% men have a chance of heart disease
# on average a person has (75 + 45)/2 = 60% chance


# In[27]:


pd.crosstab(df.target, df.sex).plot(kind="bar",
                                    figsize = (10, 6),
                                    color = ["salmon", "lightblue"]);
plt.title("Heart disease frequency by sex")
plt.xlabel("0 = No disease, 1 = Disease")
plt.ylabel("Number of male/female")
plt.legend(["Female", "Male"]);
plt.xticks(rotation=0);


# ##### Age vs Max heart rate (thalach) for heart disease

# In[31]:


plt.figure(figsize=(10, 6))

# Scatter with positive examples
plt.scatter(df.age[df.target==1], 
            df.thalach[df.target==1],
            c = 'salmon');

# Scatter with negative examples
plt.scatter(df.age[df.target==0],
            df.thalach[df.target==0],
            c = 'lightblue');

# Add info
plt.title("Heart disease in function of Age and Max heart rate")
plt.xlabel("Age")
plt.ylabel("Max heart rate")
plt.legend(["Disease", "No disease"]);


# In[32]:


# Check distribution(spread) of the age column with histogram
df.age.plot.hist();


# ##### Comparing chest pain type (cp) and heart disease frequency

# In[33]:


pd.crosstab(df.cp, df.target)


# In[37]:


pd.crosstab(df.cp, df.target).plot(kind='bar',
                                   figsize=(10, 6),
                                   color=['salmon', 'lightblue']);

plt.title("Heart disease frequency per Chest pain type")
plt.xlabel("Chest pain type")
plt.ylabel("Frequency")
plt.legend(["No disease", "Disease"]);
plt.xticks(rotation=0);


# In[38]:


# Make a correaltion matrix
df.corr()


# In[40]:


corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_matrix,
                 annot = True,
                 linewidths = 0.5,
                 fmt = ".2f",
                 cmap = "YlGnBu");


# ### 5. Modelling

# In[41]:


# Split the data into X and y
X = df.drop("target", axis=1)
y = df["target"]

# Split data into train and test sets
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[42]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[43]:


# Use scikit-learn machine learning map to find the desired classification model


# 1. Logistic Regression
# 2. K-Nearest Neighbours Classifier
# 3. Random Forest Classifier

# In[46]:


# Put models in a dictionary
models = {"Logistic Regression" : LogisticRegression(),
          "KNN" : KNeighborsClassifier(),
          "Random Forest" : RandomForestClassifier()}

# Create a function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    np.random.seed(42)
    model_scores = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[model_name] = model.score(X_test, y_test)
        
    return model_scores                                                    


# In[47]:


model_scores = fit_and_score(models = models,
                            X_train = X_train,
                            X_test = X_test,
                            y_train = y_train,
                            y_test = y_test)
model_scores


# ##### Model Comparison

# In[48]:


model_compare = pd.DataFrame(model_scores, index=["accuracy"])


# In[49]:


model_compare


# In[50]:


model_compare.T


# In[52]:


model_compare.T.plot.bar();


# In[53]:


# We are getting about 88% accuracy on logistic regression model, but we want atleast 95%z


# In[54]:


# We have got our baseline model, lets improve it


# ###### Lets look at the following :
# * Hyperparameter tuning
# * Feature importance
# * Confusion matrix
# * Cross-validation
# * Precision
# * Recall
# * F1 score
# * Classification report
# * ROC curve
# * AUC

# ###### Hyperparameter tuning by hand

# In[57]:


# let's tune KNN

train_scores = []
test_scores = []

# Create a list of different values of n_neighbors
neighbors = range(1, 21)

# Setup KNN instance
knn = KNeighborsClassifier()

# Loop through different n_neighbors
for i in neighbors:
    knn.set_params(n_neighbors = i)
    
    knn.fit(X_train, y_train)
    
    train_scores.append(knn.score(X_train, y_train))
    
    test_scores.append(knn.score(X_test, y_test))


# In[58]:


train_scores


# In[59]:


test_scores


# In[62]:


plt.plot(neighbors, train_scores, label = "Train score")
plt.plot(neighbors, test_scores, label = "Test score")
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()

print(f"Maximum KNN score on the test data : {max(test_scores)*100:.2f}%")


# In[63]:


# n_neighbors value of 11 performs best on test data (75%)
# but its accuracy is still below other models like logistic regression etc


# ###### Hyperparameter tuning with RandomizedSearchCV

# In[66]:


# Create a hyperparameter grid for LogisticRegression
log_reg_grid = { "C" : np.logspace(-4, 4, 20),
                "solver" : ["liblinear"]}

# Create a hyperparameter grid for RandomForestClassifier
rf_grid = {"n_estimators" : np.arange(10, 1000, 50),
           "max_depth" : [None, 3, 5, 10],
           "min_samples_split" : np.arange(2, 20, 2),
           "min_samples_leaf" : np.arange(1, 20, 2)}


# In[68]:


# Tune LogisticRegression
np.random.seed(42)

rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions = log_reg_grid,
                                cv = 5,
                                n_iter = 20,
                                verbose = True)

rs_log_reg.fit(X_train, y_train)


# In[69]:


rs_log_reg.best_params_


# In[70]:


rs_log_reg.score(X_test, y_test)


# In[72]:


# Tune RandomForestClassifier


# In[73]:


np.random.seed(42)

rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions = rf_grid,
                           cv = 5,
                           n_iter = 20,
                           verbose = True)
rs_rf.fit(X_train, y_train)


# In[74]:


rs_rf.best_params_


# In[75]:


rs_rf.score(X_test, y_test)


# In[76]:


# it is better than before
# But logistic regression is giving better accuracy


# ###### Hyperparameter tuning using GridSearchCV

# In[77]:


# Create a hyperparameter grid for LogisticRegression
log_reg_grid = { "C" : np.logspace(-4, 4, 30),
                "solver" : ["liblinear"]}

gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid = log_reg_grid,
                          cv = 5,
                          verbose = True)

gs_log_reg.fit(X_train, y_train)


# In[78]:


gs_log_reg.best_params_


# In[79]:


gs_log_reg.score(X_test, y_test)


# In[80]:


y_preds = gs_log_reg.predict(X_test)


# In[81]:


y_preds


# In[83]:


plot_roc_curve(gs_log_reg, X_test, y_test);


# In[84]:


print(confusion_matrix(y_test, y_preds))


# In[85]:


sns.set(font_scale = 1.5)

def plot_conf_mat(y_test, y_preds):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                    annot = True,
                    cbar = False)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    
plot_conf_mat(y_test, y_preds)


# In[86]:


print(classification_report(y_test, y_preds))


# In[87]:


gs_log_reg.best_params_


# In[88]:


clf = LogisticRegression(C = 0.20433597178569418,
                         solver = 'liblinear')


# In[93]:


# Cross validated accuracy
cv_acc = cross_val_score(clf,
                         X,
                         y,
                         cv = 5,
                         scoring = "accuracy")
cv_acc


# In[94]:


cv_acc = np.mean(cv_acc)
cv_acc


# In[96]:


# Cross validated precision
cv_precision = cross_val_score(clf,
                         X,
                         y,
                         cv = 5,
                         scoring = "precision")
cv_precision


# In[97]:


cv_precision = np.mean(cv_precision)
cv_precision


# In[98]:


# Cross validated recall
cv_recall = cross_val_score(clf,
                         X,
                         y,
                         cv = 5,
                         scoring = "recall")
cv_recall


# In[99]:


cv_recall = np.mean(cv_recall)
cv_recall


# In[100]:


# Cross validated f1-score
cv_f1 = cross_val_score(clf,
                         X,
                         y,
                         cv = 5,
                         scoring = "f1")
cv_f1


# In[101]:


cv_f1 = np.mean(cv_f1)
cv_f1


# In[108]:


cv_metrics = pd.DataFrame({"Accuracy" : cv_acc,
                           "Precision" : cv_precision,
                           "Recall" : cv_recall,
                           "F1-score" : cv_f1},
                          index = [0]
                          )


# In[103]:


cv_metrics


# In[109]:


cv_metrics.T


# In[111]:


cv_metrics.T.plot.bar(title = "Cross-validated classification metrics",
                      legend = False);


# ###### Feature importance
# * Which features contributed most to the outcomes of the model and how did they contribute?
# * Finding feature importance is different for each machine learning model

# In[113]:


gs_log_reg.best_params_


# In[114]:


clf = LogisticRegression(C = 0.20433597178569418,
                         solver = 'liblinear')


# In[115]:


clf.fit(X_train, y_train)


# In[116]:


# Check coef_
clf.coef_
# tells how each each feature is related to label


# In[118]:


# Match coef's of features to columns
feature_dict = dict(zip(df.columns, list(clf.coef_[0])))
feature_dict


# In[120]:


# Visualize feature importance
feature_df = pd.DataFrame(feature_dict, index=[0])


# In[121]:


feature_df


# In[122]:


feature_df.T


# In[124]:


feature_df.T.plot.bar(title = "Feature Importance", legend = False);


# In[125]:


pd.crosstab(df["sex"], df["target"])


# In[126]:


pd.crosstab(df["slope"], df["target"])


# In[ ]:




