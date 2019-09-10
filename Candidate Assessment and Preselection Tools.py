#!/usr/bin/env python
# coding: utf-8

# In[92]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from  sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, accuracy_score, recall_score
from sklearn.feature_selection import SelectKBest,chi2, RFE
from sklearn.utils import resample


# In[233]:


def filter_data(data):
    if len(data[data.isnull().any(axis=1)]) >= 1:
        print('Null Values exist in column(s)')
        
    else:
        print('Proceed')
        return data
    #filter code here    
   


# In[234]:


def onehot(filtered_data):
    data_num = filtered_data[['AGE','SENIORITY','NET_SALES_AMT','COMMISSION_AMT','ORDER_NUM']]
    onehot_encoder = OneHotEncoder(sparse=False,handle_unknown='ignore')
    encoded_sign = onehot_encoder.fit_transform(np.array(filtered_data['SIGN']).reshape(-1,1))
    encoded_prov = onehot_encoder.fit_transform(np.array(filtered_data['BIRTH_PROV']).reshape(-1,1))
    encoded_major = onehot_encoder.fit_transform(np.array(filtered_data['BIG_MAJOR_NAME']).reshape(-1,1))
    encoded_married = onehot_encoder.fit_transform(np.array(filtered_data['MARRIED']).reshape(-1,1))
    df_prov = pd.DataFrame(encoded_prov, columns = ['prov_' + str(i) for i in range(np.shape(encoded_prov)[1])])
    df_sign = pd.DataFrame(encoded_sign,columns=['sign' + str(i) for i in range(np.shape(encoded_sign)[1])])
    df_major = pd.DataFrame(encoded_sign,columns=['major' + str(i) for i in range(np.shape(encoded_major)[1])])
    df_married = pd.DataFrame(encoded_married, columns = ['marr_' + str(i) for i in range(np.shape(encoded_married)[1])])
    onehot_data = pd.concat([data_num,df_prov,df_sign,df_major,df_married], axis = 1)
    return onehot_data


# In[235]:


def feature_labels(onehot_data):
    sales_mean = onehot_data['NET_SALES_AMT'].mean()
    comm_mean = onehot_data['COMMISSION_AMT'].mean()
    order_mean = onehot_data['ORDER_NUM'].mean()
    onehot_data['top'] = np.where((onehot_data['NET_SALES_AMT'] > sales_mean ) & (onehot_data['COMMISSION_AMT'] > comm_mean) & (onehot_data['ORDER_NUM'] > order_mean),1,0)
    onehot_data = onehot_data.drop(columns=['NET_SALES_AMT','COMMISSION_AMT','ORDER_NUM'])
    target = np.reshape(onehot_data.loc[:,['top']].values,-1,1)
    feature = onehot_data.iloc[:,0:len(onehot_data.columns)-1].values
    return feature, target
    


# In[236]:


def feature_select(feature, label):
    #X_new = SelectKBest(score_func=chi2, k=20).fit_transform(feature, label)
    X_new = RFE(estimator = RandomForestClassifier(n_estimators=20,criterion='gini',
            max_depth= 2,random_state=0), n_features_to_select = 30).fit_transform(feature, label)
    return X_new
    


# In[237]:


def up_sample(feature_reduc, label):
    x_train, x_test, y_train, y_test = train_test_split(feature_reduc,label,test_size=0.2, random_state=0)
    X = pd.concat([pd.DataFrame(x_train), pd.DataFrame(y_train, columns=['tar'])], axis=1)
    not_top = X[X.tar==0]
    top = X[X.tar==1]
    top_upsampled = resample(top,
                          replace=True, # sample with replacement
                         n_samples=len(not_top), # match number in majority class
                          random_state=27) # reproducible results
    upsampled = pd.concat([not_top, top_upsampled])
    up_feature = upsampled.iloc[:,0:len(upsampled.columns)-1].values
    up_label = np.reshape(upsampled.iloc[:,len(upsampled.columns)-1:len(upsampled.columns)].values,-1,1)
    return up_feature, up_label


# In[238]:


def adaboost(up_feature, up_label):
    adaclassifier = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=1),n_estimators = 50)
    x_train, x_test, y_train, y_test = train_test_split(up_feature, up_label,test_size=0.2, random_state=0)
    adaclassifier.fit(x_train,y_train)
    
    print('confusion matrix',  confusion_matrix(y_test,adaclassifier.predict(x_test)))
    print( classification_report(y_test,adaclassifier.predict(x_test)))
    score = cross_val_score(adaclassifier,up_feature, up_label,cv = 10)
    print('adaboost:',score.mean())
    
    


# In[239]:


def gradboost(up_feature, up_label):        
    gradboostclassifier = GradientBoostingClassifier(loss = "deviance",learning_rate = 0.05, n_estimators = 150, subsample = 1 )
    x_train, x_test, y_train, y_test = train_test_split(up_feature, up_label,test_size=0.2, random_state=0)
    gradboostclassifier.fit(x_train,y_train)
    print('train_accuracy',accuracy_score(y_train,gradboostclassifier.predict(x_train)))
    print('test_accuracy',accuracy_score(y_test,gradboostclassifier.predict(x_test)))
    print('confusion matrix', confusion_matrix(y_test,gradboostclassifier.predict(x_test)))
    print( classification_report(y_test,gradboostclassifier.predict(x_test)))
    score = cross_val_score(gradboostclassifier,up_feature, up_label,cv=10)
    print('gradient boosting cross val:', score.mean())
    probs = gradboostclassifier.predict_proba(x_test)
    probs = probs[:, 1]
    return probs, y_test


# In[240]:


def rand_forest(up_feature, up_label):
    x_train, x_test, y_train, y_test = train_test_split(up_feature, up_label,test_size=0.2, random_state=0)
    rand_forest = RandomForestClassifier(n_estimators=100, max_depth=10,
                  max_features='auto',criterion = 'gini', random_state=1)
    rand_forest.fit(x_train,y_train)
    print('train_accuracy',accuracy_score(y_train,rand_forest.predict(x_train)))
    print('test_accuracy',accuracy_score(y_test,rand_forest.predict(x_test)))
    print('confsion matrix', confusion_matrix(y_test,rand_forest.predict(x_test)))
    print(classification_report(y_test,rand_forest.predict(x_test)))
    score = cross_val_score(rand_forest,up_feature,up_label, cv =10)
    print('random forest cross val',score.mean())
    probs = rand_forest.predict_proba(x_test)
    probs = probs[:, 1]
    return probs, y_test
   


# In[241]:


def roc_curve_auc(frst_y_probs,frst_y_test,grad_y_probs, grad_y_test):
    auc_grad = roc_auc_score(grad_y_test, grad_y_probs)
    print('gradient boosting auc:', auc_grad)
    auc_frst = roc_auc_score(frst_y_test, frst_y_probs)
    print('random forest auc:', auc_frst)
    fal_pos, tru_pos, threshold = roc_curve(grad_y_test, grad_y_probs)
    fal_pos1, tru_pos1, threshold1 = roc_curve(frst_y_test, frst_y_probs)
    plt.plot([0,1],[0,1],linestyle = '--')
    plt.plot(fal_pos, tru_pos, marker = 's',label = 'gradient boosting')
    plt.plot(fal_pos1, tru_pos1, marker = '*', label = 'random forest')
    


# In[242]:


def main():
    data = pd.read_csv('C:\\Users\\jeffr\\Python\\recommendation\\promoter_01.csv')
    data = data.drop(columns='Unnamed: 0')
    data['BIRTH_PROV'] = np.where(data['BIRTH_PROV'].isna(),'湖南省',data['BIRTH_PROV'])
    data['COMMISSION_AMT'] = np.where(data['COMMISSION_AMT']<0, 0, data['COMMISSION_AMT'])
    data['NET_SALES_AMT'] = np.where(data['NET_SALES_AMT']<0, 0, data['NET_SALES_AMT'])
    filtered_data = filter_data(data)
    onehot_data = onehot(filtered_data)
    feature, label = feature_labels(onehot_data)
    feature_reduc = feature_select(feature, label)
    up_feature, up_label = up_sample(feature_reduc, label)
    print('Gradient Boosting')
    grad_y_probs, grad_y_test = gradboost(up_feature, up_label)
    print('Random Forest')
    frst_y_probs,frst_y_test =  rand_forest(up_feature, up_label)
    print('Receiver operating characteristic curve, AUC')
    roc_curve_auc(frst_y_probs,frst_y_test,grad_y_probs, grad_y_test)


# In[243]:


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




