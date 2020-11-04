#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation
# 
# 
# ## Objective (Task 2): Predict the optimum number of clusters and represent it visually, for the given Iris dataset.
# 
# 
# 
# ### Submitted by: Sachin Shastri

# In[3]:


#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[4]:


#Load Iris dataset
iris=pd.read_csv('/Users/ankitshastri/Downloads/Iris.csv')


# In[5]:


iris.head()


# In[6]:


iris.isna()


# # Workflow: 
# Step 1- We assign an arbitrary value of 'k'(number of categories of target variable under which the attributes of data are to be classified) and calculate cluster centre.
# 
# Step 2- We calculate and plot the error between predicted values and actual values, for different values of 'k' & subsequently plot the elbow curve, with 'k' values as abscissa & respective error values as ordinate. The point where this curve has its elbow is the point where k value is optimal.
# 
# Step 3- Use the optimal value of 'k' from elbow curve to classify the attributes in said categories.

# ## Step 1:

# In[4]:


x = iris.iloc[:, [0,1,2,3]].values


# In[9]:


kmeans6 = KMeans(n_clusters=6)
y_kmeans6 = kmeans6.fit_predict(x)
print(y_kmeans6)

kmeans6.cluster_centers_


# ## Step 2:

# In[7]:


Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()


# In[7]:


#Observation: In the above plotted elbow curve, the curve has its elbow for values of 'k' between 2 & 4.
#Hence, the optimal value of 'k' should be between 2 & 4 ie 3.


# ## Step 3:

# In[10]:


kmeans3 = KMeans(n_clusters=3)
y_kmeans3 = kmeans3.fit_predict(x)
print(y_kmeans3)

kmeans3.cluster_centers_


# In[11]:


plt.scatter(x[:,0],x[:,1], c=y_kmeans3, cmap='rainbow')


# ### Conclusion: In the above plot, the three colours represent the distribution of datapoints into three categories. 
# ### Since both, the elbow curve and scatter plot suggest the value of 'k' to be 3. It can be said that, model has predicted the optimal value of 'k' correctly.

# In[ ]:




