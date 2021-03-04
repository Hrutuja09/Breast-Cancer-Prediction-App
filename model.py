#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import pickle
warnings.filterwarnings('ignore')


# In[6]:


cancer=pd.read_csv("C:\\Users\\Lenovo\\downloads\\data.csv")


# In[7]:


cancer.shape


# In[8]:


cancer.head(5)


# ## Data Wrangling

# In[9]:


cancer.isnull()


# In[10]:


cancer.isnull().sum()


# In[11]:


cancer.drop(["id", "Unnamed: 32"],axis=1, inplace=True)


# In[12]:


cancer.head(5)


# In[13]:


pd.get_dummies(cancer['diagnosis'])


# In[14]:


detection=pd.get_dummies(cancer['diagnosis'], drop_first=True)


# In[15]:


detection.head(2)


# In[16]:


cancer=pd.concat([cancer , detection], axis=1)


# In[17]:


cancer.head(5)


# In[18]:


cancer.drop(['diagnosis'],inplace=True, axis=1)


# In[19]:


cancer.head(4)


# ## Data Visualization

# In[21]:


sns.set_style("whitegrid")
sns.countplot(data=cancer, x='M', palette= 'hls')


# In[22]:


sns.boxplot(x= "M",y= "radius_mean", data= cancer)


# In[23]:


cancer["area_mean"].plot.hist(figsize= (10,5),color= "pink")


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


# In[31]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ## Logistic Regression

# In[32]:


from sklearn.linear_model import LogisticRegression


# In[33]:


reg= LogisticRegression( )


# In[34]:


reg.fit(X_train, y_train)


# In[35]:


predictions= reg.predict(X_test)


# In[36]:


from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))


# In[37]:


from sklearn.metrics import confusion_matrix


# In[38]:


confusion_matrix(y_test, predictions)


# In[39]:


from sklearn.metrics import accuracy_score


# In[40]:


accuracy_score(y_test, predictions)


# In[41]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(reg,X_test,y_test)
plt.show()


# In[63]:


pickle.dump(reg, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(model)


# ## Random Forest

# In[64]:


from sklearn.ensemble import RandomForestClassifier


# In[65]:


classifier= RandomForestClassifier(n_estimators = 80,criterion ='gini',random_state=0)


# In[66]:


classifier.fit(X_train, y_train)


# In[67]:


predictions= classifier.predict(X_test)


# In[68]:


from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))


# In[69]:


from sklearn.metrics import confusion_matrix


# In[70]:


confusion_matrix(y_test, predictions)


# In[71]:


from sklearn.metrics import accuracy_score


# In[72]:


accuracy_score(y_test, predictions)


# In[51]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(classifier,X_test,y_test)
plt.show()


# In[73]:


pickle.dump(classifier, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(model)


# # SVM

# In[52]:


from sklearn.svm import SVC


# In[53]:


SVM=SVC(kernel='linear')


# In[54]:


SVM.fit(X_train, y_train)


# In[55]:


prediction =SVM.predict(X_test)


# In[56]:


from sklearn.metrics import classification_report

print(classification_report(y_test, prediction))


# In[57]:


from sklearn.metrics import confusion_matrix


# In[58]:


confusion_matrix(y_test, prediction)


# In[59]:


from sklearn.metrics import accuracy_score


# In[60]:


accuracy_score(y_test, prediction)


# In[61]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(SVM,X_test,y_test)
plt.show()


# In[74]:


pickle.dump(SVM, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(model)


# In[ ]:





# In[ ]:




