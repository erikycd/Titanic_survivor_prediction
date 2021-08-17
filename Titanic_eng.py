# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 20:30:30 2021

@author: Erik
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#%% Importing data from CSV file

df_train_original = pd.read_csv("C:/Users/Erik/Documents/Python ML/Titanic/train.csv")
df_test_original = pd.read_csv("C:/Users/Erik/Documents/Python ML/Titanic/test.csv")

df_train = df_train_original
df_test = df_test_original

#%% Printing info

print(df_test.head)
print(df_train.head)

print('Full size data:')
print(df_test.shape)
print(df_train.shape)

print('Datatype:')
print(df_test.info())
print(df_train.info())

print('Floating data:')
print(pd.isnull(df_test).sum())
print(pd.isnull(df_train).sum())

print('Dataset statistics:')
print(df_test.describe())
print(df_train.describe())


#%% First graphics:

def bar_chart_survivor(dataset,feature):
    survived = dataset[dataset['Survived'] == 1][feature].value_counts()
    dead = dataset[dataset['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind = 'bar', stacked = True, figsize = (10,5))
    
    
def bar_chart(dataset,feature_1,feature_2):
    count_feature_1 = np.unique(dataset[feature_1].values)
    count_feature_2 = np.unique(dataset[feature_2].values)
    class_F = pd.DataFrame({'NaN' : []})
        
    for x in count_feature_1:
        
        class_1 = dataset[dataset[feature_1] == x][feature_2].value_counts()
        class_1 = class_1.sort_index()
        
        class_F = pd.concat([class_F, class_1], axis=1)
        
    class_F = class_F.drop('NaN', axis = 1)
    class_F.columns = count_feature_1
    class_F.plot(kind = 'bar', stacked = True, figsize = (10,5))
    plt.xlabel(feature_2)
    plt.ylabel(feature_1)
    
bar_chart_survivor(df_train_original,'Sex')
bar_chart(df_train_original,'Sex','Survived')


#%% Feature correlation

correlation = df_train.corr(method = 'pearson')
plt.matshow(correlation)
res = sn.heatmap(correlation, annot=True, fmt='.2f', cbar=False)
plt.show()

# Plot for HTML:
#correlacion.style.background_gradient(cmap='coolwarm').set_precision(2)

pd.plotting.scatter_matrix(df_train, figsize=(12, 8))

#%% Preprocessing 1. categorical_to_num in both training and testing sets

# Changing data type to numbers (Sex feature)
df_test['Sex'].replace(['female','male'],[0,1], inplace = True)
df_train['Sex'].replace(['female','male'],[0,1], inplace = True)

# Replacing data type to numbers (Embarked feature)
df_test['Embarked'].replace(['Q','S','C'],[1,2,3], inplace = True)
df_train['Embarked'].replace(['Q','S','C'],[1,2,3], inplace = True)


#%% Preprocessing 2. Estimating missing data (Ages)

# Replacing missing Ages by the mean Age class. Train
personas_clase_1 = df_train[df_train['Pclass'] == 1]
personas_clase_2 = df_train[df_train['Pclass'] == 2]
personas_clase_3 = df_train[df_train['Pclass'] == 3]

personas_clase_1['Age'].replace(np.nan,personas_clase_1['Age'].mean(),inplace = True)
personas_clase_2['Age'].replace(np.nan,personas_clase_2['Age'].mean(),inplace = True)
personas_clase_3['Age'].replace(np.nan,personas_clase_3['Age'].mean(),inplace = True)

df_train = pd.concat([personas_clase_1,personas_clase_2,personas_clase_3], axis=0)
df_train = df_train.sort_values(by=['PassengerId'])

# Replacing missing Ages by the mean Age class. Test
personas_clase_1 = df_test[df_test['Pclass'] == 1]
personas_clase_2 = df_test[df_test['Pclass'] == 2]
personas_clase_3 = df_test[df_test['Pclass'] == 3]

personas_clase_1['Age'].replace(np.nan,personas_clase_1['Age'].mean(),inplace = True)
personas_clase_2['Age'].replace(np.nan,personas_clase_2['Age'].mean(),inplace = True)
personas_clase_3['Age'].replace(np.nan,personas_clase_3['Age'].mean(),inplace = True)

df_test = pd.concat([personas_clase_1,personas_clase_2,personas_clase_3], axis=0)
df_test = df_test.sort_values(by=['PassengerId'])

# Age clustering in ranges
bins_age = [0,10,20,30,40,50,60,70,80]
age_class = ['0','1','2','3','4','5','6','7']
df_test['Age'] = pd.cut(df_test['Age'], bins_age, labels = age_class)
df_train['Age'] = pd.cut(df_train['Age'], bins_age, labels = age_class)

#%% Preprocessing 3. Fares
    
# Rellenar datos faltantes de precios(fare) en test
# Replacing missing Fares by the mean fare. Test
df_test['Fare'] = df_test['Fare'].replace(np.nan, df_test['Fare'].mean())

# Fare clustering in ranges
bins_fare = [-1,11,26,600]
fare_class = ['3','2','1']
df_test['Fare'] = pd.cut(df_test['Fare'], bins_fare, labels = fare_class)
df_train['Fare'] = pd.cut(df_train['Fare'], bins_fare, labels = fare_class)

#%% Preprocessing 4. Data cleaning

# Deleting Cabin columns
df_test = df_test.drop(['Cabin'], axis=1)
df_train = df_train.drop(['Cabin'], axis=1)

# Deleting Name and Ticket number columns
df_test = df_test.drop(['Name','Ticket'], axis=1)
df_train = df_train.drop(['PassengerId','Name','Ticket'], axis=1)

# Deleting rows with missing data if there is any after completing the preprocessing stage
df_train = df_train.dropna(axis=0,how='any')


#%% Machine learning process

# Separating Survived column
X = np.array(df_train.drop(['Survived'], 1))
X_norm = preprocessing.normalize(X)
y = np.array(df_train['Survived'])

# Data split
X_train, X_valid, y_train, y_valid = train_test_split(X_norm, y, test_size=0.2)

#%% Automatic clasificacion

model_Gauss = GaussianNB()
model_Gauss.fit(X_train, y_train);
y_predict_Gauss = model_Gauss.predict(X_valid)
print('Gaussian Naive Bayes:')
print(model_Gauss.score(X_valid,y_valid))

model_logreg = LogisticRegression()
model_logreg.fit(X_train, y_train);
y_predict_logreg = model_logreg.predict(X_valid)
print('Logistic Regression:')
print(model_logreg.score(X_valid,y_valid))

model_svm = SVC()
model_svm.fit(X_train, y_train);
y_predict_svm = model_svm.predict(X_valid)
print('SMV Classifier:')
print(model_svm.score(X_valid,y_valid))

model_randomF = RandomForestClassifier(n_estimators=100,max_depth=5,random_state=1)
model_randomF.fit(X_train, y_train);
y_predict_randomF = model_randomF.predict(X_valid)
print('Random Forest:')
print(model_randomF.score(X_valid,y_valid))

model_dectree = DecisionTreeClassifier(criterion='entropy',random_state=0)
model_dectree.fit(X_train, y_train);
y_predict_dectree = model_dectree.predict(X_valid)
print('Decision Tree:')
print(model_dectree.score(X_valid,y_valid))

#%% Displaying results of training classification

Res_Gauss = model_Gauss.score(X_valid,y_valid)
Res_Logis = model_logreg.score(X_valid,y_valid)
Res_SVM = model_svm.score(X_valid,y_valid)
Res_RandomF = model_randomF.score(X_valid,y_valid)
Res_DesT = model_dectree.score(X_valid,y_valid)

Algorithms = ['Naive Bayes','Linear Regression','SVM','Random Forest','Decision Tree']
Results = [Res_Gauss, Res_Logis, Res_SVM, Res_RandomF, Res_DesT]
Table_results = pd.DataFrame(Results, Algorithms, columns = ['Results'])

plt.matshow(Table_results)
res = sn.heatmap(Table_results, annot=True, fmt='.3f', cbar=False)
res.set_yticklabels(res.get_yticklabels(), rotation=0)
plt.show()


#%% Inference on test data

Inference_X = np.array(df_test.drop(['PassengerId'], 1))
Inference_X = preprocessing.normalize(Inference_X)
Inference_PassengerId = np.array(df_test['PassengerId']) 

prediction_inference = model_svm.predict(Inference_X)

submission = pd.DataFrame({"PassengerId": Inference_PassengerId,
                            "Survived": prediction_inference})


#%% Saving in the correct format for submission

# submission.to_csv('submission_4.csv', index = False)

# submission = pd.read_csv('submission_4.csv')
# print(submission.head())



















