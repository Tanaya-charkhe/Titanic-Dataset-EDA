#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("titanic.csv")

df.columns = df.columns.str.strip()

print("Dataset Shape:", df.shape)
df.head()


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


# Fill Age with median
if 'Age' in df.columns:
    df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill Embarked with mode
if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Fill Fare with median
if 'Fare' in df.columns:
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Drop Cabin if exists
if 'Cabin' in df.columns:
    df.drop(columns=['Cabin'], inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

df.isnull().sum()


# In[6]:


bins = [0, 12, 18, 35, 60, 100]
labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']

df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

df.head()


# In[7]:


overall_survival = df['Survived'].mean()
print("Overall Survival Rate:", round(overall_survival * 100, 2), "%")


# In[8]:


plt.figure()
sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival Rate by Gender")
plt.show()


# In[9]:


plt.figure()
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Survival Rate by Passenger Class")
plt.show()


# In[10]:


plt.figure()
sns.barplot(x='AgeGroup', y='Survived', data=df)
plt.xticks(rotation=45)
plt.title("Survival Rate by Age Group")
plt.show()


# In[11]:


plt.figure()
sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age Distribution by Survival")
plt.show()


# #  Key Insights
# 
# 1. Females had significantly higher survival rates compared to males.
# 2. First-class passengers had the highest survival probability.
# 3. Third-class passengers had the lowest survival rate.
# 4. Children had better survival chances.
# 5. Social and economic status strongly influenced survival outcomes.
# 
# ## Conclusion
# Survival on the Titanic was strongly influenced by gender, passenger class, and age.

# In[ ]:





# In[ ]:




