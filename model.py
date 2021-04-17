#!/usr/bin/env python
# coding: utf-8

# # Introduction

# Breast cancer is a type of cancer that starts in the breast. Cancer starts when cells begin to grow out of control. Breast cancer cells usually form a tumor that can often be seen on an x-ray or felt as a lump. Breast cancer occurs almost entirely in women, but men can get breast cancer, too. It’s important to understand that most breast lumps are benign and not cancer (malignant). Non-cancerous breast tumors are abnormal growths, but they do not spread outside of the breast.
# They are not life threatening, but some types of benign breast lumps can increase a woman's risk of getting breast cancer.

# Breast cancer (BC) is one of the most common cancers among women worldwide, representing the majority of new cancer cases and cancer-related deaths according to global statistics, making it a significant public health problem in today’s society. The early diagnosis of BC can improve the prognosis and chance of survival significantly, as it can promote timely clinical treatment to patients. ML techniques are being broadly used in the breast cancer classification problem. They provide high classification accuracy and effective diagnostic capabilities.
# 
# Our main objective is to build an app using Flask APIs and deploy on Heroku to classify whether the breast cancer is benign or malignant.

# # Importing library

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import pickle
warnings.filterwarnings('ignore')


# In[2]:


cancer=pd.read_csv("C:\\Users\\Lenovo\\downloads\\data.csv")


# In[3]:


cancer.shape


# In[4]:


cancer.head(5)


# In[5]:


cancer.describe()


# In[6]:


cancer.info()


# ## Data Wrangling

# In[7]:


cancer.isnull()


# In[8]:


cancer.isnull().sum()


# In[9]:


heat_map=sns.heatmap(cancer.isnull(), yticklabels= False, cbar= True, vmin= 0,vmax= 1 )


# In[10]:


cancer.drop(["id", "Unnamed: 32"],axis=1, inplace=True)


# In[11]:


cancer.head(5)


# In[12]:


pd.get_dummies(cancer['diagnosis'])


# In[13]:


detection=pd.get_dummies(cancer['diagnosis'], drop_first=True)


# In[14]:


detection.head(2)


# In[15]:


cancer=pd.concat([cancer , detection], axis=1)


# In[16]:


cancer.head(5)


# In[17]:


cancer.drop(['diagnosis'],inplace=True, axis=1)


# In[18]:


cancer.head(4)


# In[19]:


cancer.head(25)


# ## Data Visualization

# In[20]:


sns.set_style("whitegrid")
sns.countplot(data=cancer, x='M', palette= 'hls')


# In[21]:


sns.boxplot(x= "M",y= "radius_mean", data= cancer)


# In[22]:


cancer["area_mean"].plot.hist(figsize= (10,5),color= "pink")


# In[62]:


cancer["radius_mean"].plot.hist(figsize= (10,5),color= "blue")


# # Correlation

# In[23]:


sns.heatmap(cancer.corr(),cmap="gist_earth",annot=True)


# ## Model

# In[24]:


X= cancer.drop("M", axis= 1)
y= cancer["M"]


# In[25]:


X.head()


# In[26]:


y.head()


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.3, random_state= 1)


# In[29]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ## Logistic Regression

# In[30]:


from sklearn.linear_model import LogisticRegression


# In[31]:


reg= LogisticRegression( )


# In[32]:


reg.fit(X_train, y_train)


# In[33]:


predictions= reg.predict(X_test)


# In[34]:


from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))


# In[35]:


from sklearn.metrics import confusion_matrix


# In[36]:


confusion_matrix(y_test, predictions)


# In[37]:


from sklearn.metrics import accuracy_score


# In[38]:


accuracy_score(y_test, predictions)


# In[39]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(reg,X_test,y_test)
plt.show()


# In[40]:


pickle.dump(reg, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(model)


# # SVM

# In[41]:


from sklearn.svm import SVC


# In[42]:


from sklearn.model_selection import GridSearchCV
parameters={'C': [0.1,1,10,100,1000],
           'gamma':[1,0.1,0.01,0.001,0.00001],
           'kernel':['rbf']}


# In[43]:


grid=GridSearchCV(SVC(),parameters,refit=True,verbose=5)
grid.fit(X_train, y_train)


# In[44]:


prediction=grid.predict(X_test)


# In[45]:


from sklearn.metrics import classification_report
print(classification_report(y_test, prediction))


# In[46]:


from sklearn.metrics import confusion_matrix


# In[47]:


confusion_matrix(y_test, prediction)


# In[48]:


from sklearn.metrics import accuracy_score


# In[49]:


accuracy_score(y_test, prediction)


# In[50]:


pickle.dump(grid, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(model)


# # Random forest

# In[51]:


from sklearn.ensemble import RandomForestClassifier


# In[52]:


classifier= RandomForestClassifier(n_estimators = 80,criterion ='gini',random_state=0)


# In[53]:


classifier.fit(X_train, y_train)


# In[54]:


predictions= classifier.predict(X_test)


# In[55]:


from sklearn.metrics import classification_report

print(classification_report(y_test, prediction))


# In[56]:


from sklearn.metrics import confusion_matrix


# In[57]:


confusion_matrix(y_test, predictions)


# In[58]:


from sklearn.metrics import accuracy_score


# In[59]:


accuracy_score(y_test, predictions)


# In[60]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(classifier,X_test,y_test)
plt.show()


# In[61]:


pickle.dump(classifier, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(model)


# In[ ]:




