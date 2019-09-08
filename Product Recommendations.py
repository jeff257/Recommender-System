#!/usr/bin/env python
# coding: utf-8

# In[199]:


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sqlalchemy import create_engine
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


# In[301]:


def filter_data(data_excel,q_bought,dts, dte):
    newdf = data_excel[(data_excel['Quantity'] > q_bought) & (data_excel['Order Date'] <= dte) & (data_excel['Order Date'] >= dts) ]
    newdf = newdf[['Customer ID','Product Name','Quantity']]
    newdf = newdf.drop_duplicates()
    return newdf


# In[302]:


def preparemldata(filtered_data):
    ml_data = pd.pivot_table(filtered_data,index=['Product Name'], columns=['Customer ID'], values='Quantity').fillna(0)
    #ml_data = csr_matrix(ml_data.values)
    return ml_data


# In[317]:


def dim_reduc(ml_data):
    pca = PCA()
    X_train = pca.fit_transform(ml_data)
    explained_variance = pca.explained_variance_ratio_
    comp_selection = 0
    for i in range(len(ml_data.columns)):
        if explained_variance[0:i].sum() > 0.95: # look for number of components 
            #print(i, explained_variance[0:i].sum())
            break
        comp_selection = comp_selection + 1
    pca = PCA(n_components = comp_selection)
    ml_data_f = pca.fit_transform(ml_data)
    return ml_data_f   


# In[318]:


def knn(ml_data,nb, metric):
    model_knn = NearestNeighbors(n_neighbors=nb,metric = metric,algorithm = 'auto').fit(ml_data)
    distance,indices = model_knn.kneighbors(ml_data)
    return  indices


# In[319]:


# main method
def main():
    data_excel = pd.read_excel('C:\\Users\\jeffr\\Python\\recommendation\\superstore.xls',sheetname =0)
    filtered_data = filter_data(data_excel,6,'2015-01-01','2017-12-31')
    ml_data = preparemldata(filtered_data)
    ml_data_f = dim_reduc(ml_data)
    indices = knn(ml_data_f,5,'cosine')
    recommendation = pd.DataFrame(np.array(ml_data.index[indices]))
    recommendation = recommendation.rename(columns={
        0:'items',
        1:'recommendation1',
        2:'recommendation2',
        3:'recommendation3',
        4:'recommendation4',

    })
    recommendation.to_csv('C:\\Users\\jeffr\\Python\\recommendation\\recommended_item.csv')
    print('done')


# In[320]:


if __name__ == "__main__":
    main()


# In[ ]:




