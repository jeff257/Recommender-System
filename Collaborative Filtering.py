#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sqlalchemy import create_engine
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA,KernelPCA
import pymssql


# In[67]:


def connect_db(server, db,user, pw, table):
    con = pyodbc.connect('Driver={};'
                      'Server={};'
                      'Database={};'
                      'Trusted_Connection=yes;'
                       "uid={};pwd={}" .format('SQL Server Native Client 11.0',server,db,user,pw))
    cursor = con.cursor()
    df = pd.read_sql_query('select * from {}'.format('Orders'), con)
    return df 


# In[68]:


def filter_data(data_excel,q_bought,dts, dte):
    newdf = data_excel[(data_excel['Quantity'] > q_bought) & (data_excel['Order Date'] <= dte) & (data_excel['Order Date'] >= dts) ]
    newdf = newdf[['Customer ID','Product Name','Quantity']].set_index(['Customer ID'])
    newdf = newdf[newdf.groupby(['Customer ID'])['Product Name'].nunique() > 3]
    newdf = newdf.drop_duplicates()
    return newdf


# In[69]:


def preparemldata(filtered_data):
    ml_data = pd.pivot_table(filtered_data,index=['Product Name'], columns=['Customer ID'], values='Quantity').fillna(0)
    ml_data = ml_data.T
    #ml_data = csr_matrix(ml_data.values)
    return ml_data


# In[70]:


def dim_reduc(ml_data):
    pca = PCA() #feature extraction
    pca.fit_transform(ml_data)
    explained_variance = pca.explained_variance_ratio_
    -np.sort(-explained_variance)
    comp_selection = 0
    for i in range(len(ml_data.columns)):
        if explained_variance[0:i].sum() > 0.95: # look for the number of components that can represent 95% of variance
            #print(i, explained_variance[0:i].sum())
            break
        comp_selection = comp_selection + 1
    pca = PCA(n_components = comp_selection)
    ml_data_f = pca.fit_transform(ml_data) #fit the data and transform the data
    return ml_data_f   


# In[71]:


def knn(ml_data,nb, metric):
    model_knn = NearestNeighbors(n_neighbors=nb,metric = metric,algorithm = 'auto').fit(ml_data)
    distance,indices = model_knn.kneighbors(ml_data)
    return  distance,indices


# In[72]:


def collaborative_fil(dis,indices,ml_data,num_of_recom):
    newdf = pd.DataFrame()
    for user in range(len(ml_data)):
        user_arr = np.array(ml_data.index[indices])[user][1:num_of_recom]
        user_pref = ml_data[ml_data.index.isin(user_arr)]
        user_pref = user_pref.T
        srs_name = pd.Series(np.array(ml_data.index[indices])[user][0:1])
        srs = pd.Series([])
        for i in range(num_of_recom-1):
            srs = srs.append(user_pref.nlargest(5,user_pref.columns[i])[user_pref.columns[i]]*dis[user][i+1])
            srs_name = srs_name.append(pd.Series(np.array(ml_data.index[indices])[user][0:1]))
            
        srs = srs[srs>0].sort_values(ascending = True)[0:5] 
        newdf = newdf.append(pd.concat([srs_name.reset_index(drop=True),srs.reset_index()],axis=1))
    return newdf


# In[73]:


def main():
    db_table = connect_db('localhost','AdventureWorksDW2012','jeffrey','admin','Orders')
    #data_excel = pd.read_excel('C:\\Users\\jeffr\\Python\\recommendation\\superstore.xls',sheetname =0)
    filtered_data = filter_data(db_table,6,'2015-01-01','2017-12-31')
    ml_data = preparemldata(filtered_data)
    ml_data_f = dim_reduc(ml_data)
    dis, indices = knn(ml_data_f,5,'cosine')
    cf_result = collaborative_fil(dis,indices,ml_data,num_of_recom = 5)
    cf_result = cf_result.rename(columns={
        0:'users',
        'index':'recommendation',
        0:'score'
        

    })
    cf_result.to_csv('C:\\Users\\jeffr\\Python\\recommendation\\cf_recommended_item.csv')
    print('done')


# In[74]:


if __name__ == "__main__":
    main()


# In[ ]:




