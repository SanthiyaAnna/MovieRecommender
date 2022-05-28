#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np 
import pandas as pd


# In[ ]:





# In[5]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[23]:


movies.head(2)


# In[24]:


credits.head(1)['crew'].values


# In[27]:


credits.head(1)


# In[28]:


movies=movies.merge(credits,on='title')
movies.head(1)


# In[29]:


movies.shape


# 

# In[30]:


credits.shape


# In[31]:


# genres
# id
# keywords
# title
# overview
# cast
# crew

movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[32]:


movies.info()


# In[13]:


movies.head()


# In[33]:


import ast
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 


# In[ ]:





# In[14]:


movies.isnull().sum()


# In[34]:


movies.dropna(inplace=True)


# In[35]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[36]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[18]:


# '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
# ['Action','Adventure','Fantasy','Scifi']


# In[37]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[38]:


def convert3(text):
    L=[]
    counter=0
    for i in ast.literal_eval(text):
        if counter <3:
            L.append(i['name'])
            counter+=1
    return L


# In[39]:


movies['cast'] = movies['cast'].apply(convert)
movies.head()


# In[40]:


movies['cast'] = movies['cast'].apply(lambda x:x[0:3])


# In[41]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[42]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[44]:


#movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.sample(5)


# In[45]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[47]:


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[48]:


movies.head()


# In[49]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[50]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[51]:


new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
#new.head()


# In[52]:


new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()


# In[53]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[54]:


vector = cv.fit_transform(new['tags']).toarray()


# In[55]:


vector.shape


# In[56]:


from sklearn.metrics.pairwise import cosine_similarity


# In[57]:


similarity = cosine_similarity(vector)


# In[58]:


similarity


# In[59]:


new[new['title'] == 'The Lego Movie'].index[0]


# In[60]:


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)
        


# In[68]:


recommend('Gandhi')


# In[62]:


import pickle


# In[81]:


pickle.dump(new,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




