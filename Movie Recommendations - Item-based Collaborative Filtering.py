#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sqlalchemy import create_engine
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA,KernelPCA


# In[3]:


def preparemldata(data_movie): # return ml_data as the original one which can be used to evaluate the model
    ml_data = pd.pivot_table(data_movie,index=['Title'], columns=['User Id'], values='Rating')
    print('Sparcity:',(ml_data.size - np.count_nonzero(ml_data.fillna(0)))/ml_data.size)
    ml_data2 = ml_data.apply(lambda row: row.fillna(row.mean()), axis=0)
    return ml_data,ml_data2


# In[4]:


def dim_reduc(ml_data):
    pca = PCA() #feature extraction
    pca.fit_transform(ml_data)
    explained_variance = pca.explained_variance_ratio_
    -np.sort(-explained_variance)
    comp_selection = 0
    for i in range(len(ml_data.columns)):
        if explained_variance[0:i].sum() > 0.95: # look for number of components 
            print(i, explained_variance[0:i].sum())
            break
        comp_selection = comp_selection + 1
    pca = PCA(n_components = comp_selection)
    ml_data_f = pca.fit_transform(ml_data) #fit the data and transform the data
    return ml_data_f  


# In[5]:


def knn(ml_data_reduc, metric):
    model_knn = NearestNeighbors(n_neighbors=len(ml_data_reduc),metric = metric,algorithm = 'auto').fit(ml_data_reduc)
    distance,indices = model_knn.kneighbors(ml_data_reduc)
    return  distance,indices


# In[6]:


def item_based_cf(ml_data,distance,indices):
    prd = pd.DataFrame(index = [1])
    test_prd = pd.DataFrame(index = [1])
    for user in range(len(ml_data.columns)):
        df = ml_data.iloc[:,user].reset_index() # show index for movies 
        df.columns = ['product','rating']
        dis_index = df[df['rating'].notnull()].index
        score =  np.array(ml_data.iloc[:,user][ml_data.iloc[:,user].notnull()]) #the score the user give to the product
        reverse = [i[::-1] for i in distance]
        rev_dis = np.array(reverse)
        w_score = rev_dis[dis_index] * np.array(ml_data.iloc[:,user][ml_data.iloc[:,user].notnull()]).reshape((len(score), 1))

        w_score_multi = w_score[0]
        num = 0
        numerator = rev_dis[dis_index][0]
        while num + 1 < len(w_score): # add all arrays together as well as the weights to calculate the weighted sum 
            w_score_multi = w_score_multi + w_score[num + 1]
            numerator = numerator + rev_dis[dis_index][num + 1]
            num = num + 1 
        prd = prd.merge(pd.DataFrame(w_score_multi/numerator,columns = [ml_data.columns[user]]),how = 'right',left_index=True,right_index=True)
        # construct a dataframe for prodicted rating of what a user has watched, excluding the blank rating which we filled by average
        test_prd_temp = prd[ml_data.columns[user]].copy()
        test_prd_temp[test_prd_temp.index.isin(dis_index) == False] = 0
        test_prd = test_prd.merge(pd.DataFrame(test_prd_temp),how = 'right',left_index=True,right_index=True)
    return prd, test_prd


# In[35]:


def recommended_movies(prd,ml_data2,data_movie, test_prd,ml_data,num_recom):
    df_prd = prd.merge(ml_data2.reset_index().iloc[:,0:1],left_index=True,right_index= True) # find the movie titles
    lstcol  = list(df_prd.columns)
    lstcol.insert(0,lstcol[-1])
    del lstcol[-1]
    lstcol
    df_prd = df_prd[lstcol].set_index(['Title'])
    #evaluation metrics
    eval(ml_data,test_prd,df_prd,data_movie,num_recom)
    unpivot_prd_table = df_prd.unstack().reset_index(name='predicted').rename(columns= {
    'level_0':'User ID'
    })
    result = data_movie.merge(unpivot_prd_table,left_on='User Id', right_on='User ID', how='right')
    final = result[result['Title_x']!=result['Title_y']][['User Id','Title_x','Title_y','Rating','predicted']]
    final.sort_values(['User Id', 'Title_x']).to_csv('C:\\Users\\jeffr\\Python\\recommendation\\item_based_recommended_movies.csv')
    


# In[54]:


def eval(ml_data,test_prd,df_prd,data_movie,num_recom):
    recall_arr = []
    percision_arr = []
    #calculate recall & percision
    for user_id in df_prd.columns:
        top_recom = df_prd.nlargest(num_recom,user_id)[[user_id]] # number of movies to recommend 
        top_recom = top_recom.rename(columns={
        user_id : 'prd_rating'
        })
        relevant = data_movie[(data_movie['User Id'] == user_id) & (data_movie['Rating'] >= data_movie[['User Id','Rating']].groupby(['User Id']).mean().loc[user_id].max())]
        recall_percision_tab = top_recom.reset_index().merge(relevant, left_on ='Title',right_on = 'Title', how = 'right')  
        recall = recall_percision_tab['prd_rating'].count()/recall_percision_tab['Movie Id'].count()
        percision =  recall_percision_tab['prd_rating'].count()/ num_recom # same as  the variable in n largest 
        recall_arr.append(recall)
        percision_arr.append(percision)
    print('Recall:',np.mean(recall_arr))
    print('Percision:',np.mean(percision_arr))
    #MSE, RMSE # subtract the original dataframe from the one with predicted rating, remember, both are evaluated based on what a user really watched
    ml_data = ml_data.reset_index(drop=True).fillna(0)
    mse = (np.square(ml_data-test_prd).sum().sum())/df_prd.size
    rmse = np.sqrt(mse)
    #MAE
    mae = (np.absolute(ml_data-test_prd).sum().sum())/df_prd.size
    print('RMSE:', rmse)
    print('MAE',mae)


# In[55]:


def main():
    data_movie = pd.read_csv('C:\\Users\\jeffr\\Python\\recommendation\\movies_rating.csv')
    data_movie['year'] = data_movie['Title'].str.extract('\(([0-9]+)\)')
    data_movie = data_movie[data_movie['year'].isnull() == False]
    data_movie['year'] = data_movie['year'].astype('int64')
    data_movie = data_movie[data_movie['year'] > 2013]
    ml_data,ml_data2 = preparemldata(data_movie)
    ml_data_reduc = dim_reduc(ml_data2)
    distance,indices = knn(ml_data_reduc,'cosine')
    prd, test_prd = item_based_cf(ml_data,distance,indices)
    recommended_movies(prd,ml_data2,data_movie, test_prd,ml_data,10)


# In[56]:


if __name__ == "__main__":
    main()


# In[ ]:




