# Sarcopenia_prediction_summer_project_2022

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 

patients_df = pd.read_excel('pt characteristics.xls', index_col=None, na_values=('NA'))
patients_df['PreRT Skeletal Muscle status'].replace(['SM depleted', 'SM not depleted'], [1,0], inplace=True)
patients_df['Sex'].replace(['Female', 'Male'], [1,0], inplace=True)
patients_df['PostRT Skeletal Muscle status'].replace(['SM depleted', 'SM not depleted'], [1,0], inplace=True)

patients_df.shape
patients_df

columns = patients_df.columns
columns_subset = ['Sex', 'Age', 'PreRT Skeletal Muscle status', 'Pre-RT L3 Adipose Tissue Cross Sectional Area (cm2)', 'Current Smoker', 'PostRT Skeletal Muscle status']
dropped_columns = [column for column in columns if column not in columns_subset]

patients_df_reduced = patients_df.drop(dropped_columns, axis = 1)

patients_df_reduced

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = patients_df_reduced.iloc[:,0:5] # aforementioned chosen features
Y = patients_df_reduced.iloc[:,-1] # target output of Post RT Skeletal Muscle Status
X
Y

best_features = SelectKBest(score_func=chi2, k=1)
fit = best_features.fit(X,Y)

p_scores = pd.DataFrame(fit.scores_)
p_columns = pd.DataFrame(X.columns)

features_scores = pd.concat([p_columns, p_scores], axis=1)
features_scores.columns = ['Features', 'Score']
features_scores.sort_values(by = 'Score')

#Building Model
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.4,
                                                    random_state = 100)
logreg = LogisticRegression()
logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

print (X_test) #test dataset
print (y_pred) #predicted values

from sklearn import metrics
from sklearn.metrics import classification_report

print('Accuracy: ' ,metrics.accuracy_score(y_test, y_pred))
print('Recall: ' ,metrics.recall_score(y_test, y_pred, zero_division = 1))
print('Precision:' ,metrics.precision_score(y_test, y_pred, zero_division = 1))
print('CL Report:' ,metrics.classification_report(y_test, y_pred, zero_division = 1))

#ROC Curve
y_pred_proba = logreg.predict_proba(X_test) [::,1]

false_positive_rate, true_positive_rate, _ = metrics.roc_curve(y_test, y_pred_proba)

auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(false_positive_rate, true_positive_rate,label = 'AUC=' +str(auc))
plt.title('ROC Curve')
plt.ylabel('True Positive Rate')
plt.xlabel('false Positive Rate')
plt.legend(loc = 4)


