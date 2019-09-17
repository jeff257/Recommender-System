#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import heapq
from scipy.sparse import csr_matrix
import math
from sklearn.model_selection import cross_val_score, train_test_split


# In[45]:


movie_rating = pd.read_csv('C:\\Users\\jeffr\\Python\\recommendation\\movie_rating.csv')


# In[46]:


pivot_movie_rating = pd.pivot_table(movie_rating, index = "eth0_mac", columns = "title", values = "rating").fillna(1)


# In[47]:


number_matrix = pivot_movie_rating.values


# In[48]:


demean_number_matrix = np.mean(number_matrix,axis = 1)


# In[49]:


new_number_matrix = number_matrix  - demean_number_matrix.reshape(-1,1)


# In[50]:


U, sigma, Vt = svds(new_number_matrix, k = min(new_number_matrix.shape)-15)
diag_sigma = np.diag(sigma)
predicted_rating = np.dot(np.dot(U,diag_sigma),Vt) + demean_number_matrix.reshape(-1,1)
predicted_rating_test = pd.DataFrame(predicted_rating, columns = pivot_movie_rating.columns, index = pivot_movie_rating.index)


# In[51]:


unpivot_table = predicted_rating_test.stack().reset_index(name='predicted_rating')


# In[52]:


recommendation_table = unpivot_table.merge(movie_rating, how = "left", left_on = ['eth0_mac','title'], right_on = ['eth0_mac','title'])
recommendation = recommendation_table[recommendation_table['rating'].isnull() == True ]
recommendation['predicted_rating'] = round(recommendation.predicted_rating,1)
recommendation = recommendation.sort_values(by =['eth0_mac','predicted_rating'], ascending=False)


# In[53]:


recommendation = recommendation.sort_values(['eth0_mac','predicted_rating'], ascending = False).groupby(['eth0_mac']).head(10)


# In[54]:


recommendation.to_csv("C:\\Users\\jeffr\\Python\\recommendation\\mf_recommended_item.csv", sep='\t', encoding='utf-8')


# In[55]:


err = predicted_rating - new_number_matrix


# In[38]:


np.mean(predicted_rating - new_number_matrix)


# In[ ]:




