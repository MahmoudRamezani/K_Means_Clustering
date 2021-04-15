"""
Created on Sat Mar  6 19:16:03 2021
    In this script, we try to implement k-means clustering algorithm for the 
    college data. The aim is to find if a college is private or public. Besides, 
    we try to do some explanatory data analyses. 
@author: mahmoud.ramezani@uia.no
"""
#%% importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report
#%% Reading Data
colleges = pd.read_csv('College_Data',index_col=0)
colleges.columns
colleges.head()
colleges.loc[colleges['Apps']==np.max(colleges['Apps'])]
#%% Data Visualizations
sns.scatterplot(data=colleges,x='Outstate',y='F.Undergrad',hue='Private')
sns.scatterplot(data=colleges,x='Room.Board',y='Grad.Rate',hue='Private')
sns.set_style('whitegrid')
g= sns.FacetGrid(colleges,hue='Private',height=4,aspect = 1)
g = g.map(plt.hist,'Outstate')
plt.legend(labels=['Private','Public'])
g= sns.FacetGrid(colleges,hue='Private',height=4,aspect = 2)
g = g.map(plt.hist,'Grad.Rate')
plt.legend(labels=['Private','Public'])
colleges[colleges['Grad.Rate']>100]['Grad.Rate'] 
colleges['Grad.Rate']['Cazenovia College']=100
g= sns.FacetGrid(colleges,hue='Private',height=4,aspect = 2)
g = g.map(plt.hist,'Grad.Rate')
#%%  K-means clustering
model = KMeans(n_clusters=2)
model.fit(colleges.drop(['Private'],axis=1))
model.cluster_centers_
colleges['Cluster'] = colleges['Private'].apply(lambda x: 0 if x=='Yes' else 1)
model.labels_
np.mean(colleges['Cluster']==model.labels_)
print(confusion_matrix(colleges['Cluster'],model.labels_))
print(classification_report(colleges['Cluster'],model.labels_))