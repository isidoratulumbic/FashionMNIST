# -*- coding: utf-8 -*-
"""
Created on Wed May  3 13:27:34 2023

@author: smark
"""

#%%Ucitavanje biblioteka

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import svm

#%%Ucitavanje podataka

traindata = pd.read_csv("C:/Users/isido/Desktop/FashionMNIST/PROJEKATPO/fashion-mnist_train.csv")
testdata = pd.read_csv('C:/Users/isido/Desktop/FashionMNIST/PROJEKATPO/fashion-mnist_test.csv')

print(traindata)
print(testdata)

print('Broj obeležja je: ', traindata.shape[1])
print('Broj uzoraka je: ', traindata.shape[0])
print('Obeležja su:\n', traindata.dtypes)

#%%Podela na slike i labele

data_train=traindata.iloc[:,1:785]/ 255.0
label_train=pd.DataFrame([traindata.iloc[:,0]]).T
data_test=testdata.iloc[:,0:784]/ 255.0

print(label_train)
print(data_test)

#%%Pregled podataka

label_train.value_counts()

#%%

categoryMap={0 :'T-shirt/Top',
1 :'Trouser',
2 :'Pullover',
3 :'Dress',
4 :'Coat',
5 :'Sandal',
6 :'Shirt',
7 :'Sneaker',
8 :'Bag',
9 :'Ankle boot'}
label_train['category']=label_train['label'].map(categoryMap)

#%%
##########ispitati##############
L = 5 #5 slika u koloni
W = 6 #6 u redu
fig, axes = plt.subplots(L, W, figsize = (12,12))
axes = axes.ravel()

for i in range(30):
    axes[i].imshow(data_train.values.reshape((data_train.shape[0], 28, 28))[i], cmap=plt.get_cmap('gray'))
    axes[i].set_title("class " + str(label_train['label'][i]) + ": "+ label_train['category'][i])
    axes[i].axis('off')
plt.show()

#%%Provera nedostajucih vrednosti

print(testdata.isnull().sum())
print(traindata.isnull().sum())

#%%Podela na trening i validacione podatke 

l_train=pd.DataFrame([traindata.iloc[:,0]]).T
X_train, X_val, Y_train, Y_val = train_test_split(data_train, l_train, test_size = 0.25, random_state=255)
print(l_train)
#%%Standardizacija

np.mean(X_train.values),np.std(X_train.values),np.mean(X_val.values),np.std(X_val.values)

#%%

X_train=StandardScaler().fit_transform(X_train)
X_val=StandardScaler().fit_transform(X_val)

#%%

np.mean(X_train),np.std(X_train),np.mean(X_val),np.std(X_val)

#%%

column_name=['pixel'+str(i) for i in range(1,785)]
X_train = pd.DataFrame(X_train,columns =column_name)
X_val = pd.DataFrame(X_val,columns =column_name)
print(X_train)
#%%Redukcija dimenzija

pca = PCA(n_components=0.9,copy=True, whiten=False)
X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)
print(pca.explained_variance_ratio_)

#%%

var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
fig = go.Figure(data=go.Scatter(x=list(range(1,len(var)+1)), y=var))
fig.update_layout(title='PCA Variance Explained',
                   xaxis_title='# Of Features',
                   yaxis_title='% Variance Explained')
fig.show()

#%%

pcn=X_train.shape[1]

#%%

X_train = pd.DataFrame(X_train,columns =column_name[0:pcn])
X_val = pd.DataFrame(X_val,columns =column_name[0:pcn])

#%% OBUKA MODELA

##Gaussian Naive Bayes

start_time = time.time()
NB = GaussianNB()
NB.fit(X_train, Y_train.values.ravel())
y_train_prd = NB.predict(X_train)
y_val_prd = NB.predict(X_val)
acc_train_nb=accuracy_score(Y_train,y_train_prd )
acc_val_nb=accuracy_score(Y_val,y_val_prd)
print("tačnost na trening skupu:{:.4f}\ntačnost na validacionom skupu:{:.4f}".format(acc_train_nb,
acc_val_nb))
print("--- %s sekunde ---" % (time.time() - start_time))

#%%

con_matrix = pd.crosstab(pd.Series(Y_val.values.flatten(), name='Tačno' ),pd.Series(y_val_prd, name='Predviđeno'))
plt.figure(figsize = (9,6))
plt.title("Matrica konfuzije za Gaussian Naive Bayes")
sns.heatmap(con_matrix, cmap="Greens", annot=True, fmt='g')
plt.show()

#%% 

##Random Forest Classifier

start_time = time.time()
rfc = RandomForestClassifier( random_state=0)
rfc.fit(X_train, Y_train.values.ravel())
y_train_prd = rfc.predict(X_train)
y_val_prd = rfc.predict(X_val)
acc_train_rfc=accuracy_score(Y_train,y_train_prd )
acc_val_rfc=accuracy_score(Y_val,y_val_prd)
print("tačnost na trening skupu:{:.4f}\ntačnost na validacionom skupu:{:.4f}".format(acc_train_rfc,
acc_val_rfc))
print("--- %s sekunde ---" % (time.time() - start_time))

#%%

con_matrix = pd.crosstab(pd.Series(Y_val.values.flatten(), name='Tačno' ),pd.Series(y_val_prd, name='Predviđeno'))
plt.figure(figsize = (9,6))
plt.title("Matrica konfuzije za Random Forest Classifier")
sns.heatmap(con_matrix, cmap="Greens", annot=True, fmt='g')
plt.show()

#%%

##XGBoost

start_time = time.time()
xgb = XGBClassifier()
xgb.fit(X_train, Y_train.values.ravel())
y_train_prd = xgb.predict(X_train)
y_val_prd = xgb.predict(X_val)
acc_train_xgb=accuracy_score(Y_train,y_train_prd )
acc_val_xgb=accuracy_score(Y_val,y_val_prd)
print("tačnost na trening skupu:{:.4f}\ntačnost na validacionom skupu:{:.4f}".format(acc_train_xgb,
acc_val_xgb))
print("--- %s sekunde ---" % (time.time() - start_time))

#%%

con_matrix = pd.crosstab(pd.Series(Y_val.values.flatten(), name='Tačno' ),pd.Series(y_val_prd, name='Predviđeno'))
plt.figure(figsize = (9,6))
plt.title("Matrica konfuzije za XGBoost Classifier")
sns.heatmap(con_matrix, cmap="Greens", annot=True, fmt='g')
plt.show()

#%% model comparison
acc_combine = {'Model':  ['Gaussian Naive Bayes','Random Forest Classifier','XGBoost'],
        'Accuracy_Tra': [ acc_train_nb,acc_train_rfc,acc_train_xgb],
        'Accuracy_Val': [acc_val_nb,acc_val_rfc,acc_val_xgb]
        }

#%% 
plt.switch_backend('Qt5Agg')

fig = go.Figure(data=[
    go.Bar(name='trening skup', x=acc_combine['Model'], y=acc_combine['Accuracy_Tra'],text=np.round(acc_combine['Accuracy_Tra'],2),textposition='outside' , marker=dict(color='#5DBB63')),
    go.Bar(name='validacioni skup', x=acc_combine['Model'], y=acc_combine['Accuracy_Val'],text=np.round(acc_combine['Accuracy_Val'],2),textposition='outside', marker=dict(color='#AEF359'))
])

fig.update_layout(barmode='group',title_text='Poređenje tačnosti na različitim modelima',yaxis=dict(
        title='Tačnost'))
fig.show()

fig.write_html('chart.html')
