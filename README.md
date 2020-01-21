
# Recommendation Systems


## Introduction

In this lesson, you'll investigate a very different take on networks and investigate how recommendation systems can be built off of networks. 

## Objectives

You will be able to: 
- Demonstrate how to create a simple collaborative filtering recommender system 
- Use graph-based similarity metrics to create a collaborative filtering recommender system 


## Motivating Ideas

When recommending items to a user whether they be books, music, movies, restaurants or other consumer products, one is typically trying to find the preferences of other users with similar tastes who can provide useful suggestions for the user in question. With this, examining the relationships amongst users and their previous preferences can help identify which users are most similar to each other. Alternatively, one can examine the relationships between the items themselves. These two perspectives underlying the two predominant means to recommendation systems: item-based and people-based. 

## Collaborative Filtering

One popular implementation of this intuition is collaborative filtering. This starts by constructing a matrix of user or item similarities. For example, you might calculate the distance between users based on their mutual ratings of items. From there, you then select the top $n$ similar users or items. Finally, in the case of users, you then project an anticipated rating for other unreviewed items of the user based on the preferences of these similar users. Once sorted, these projections can be then used to serve recommendations to other users.

## Importing a Dataset

To start, you'll need to import a dataset as usual. For this lesson, you'll take a look at the Movie-Lens dataset which contains movie reviews for a large number of individuals. While the dataset is exclusively older movies, it should still make for an interesting investigation.


```python
import pandas as pd
df = pd.read_csv('ml-100k/u.data', delimiter='\t', 
                 names=['user_id' , 'item_id' , 'rating' , 'timestamp'])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>item_id</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>196</td>
      <td>242</td>
      <td>3</td>
      <td>881250949</td>
    </tr>
    <tr>
      <th>1</th>
      <td>186</td>
      <td>302</td>
      <td>3</td>
      <td>891717742</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>377</td>
      <td>1</td>
      <td>878887116</td>
    </tr>
    <tr>
      <th>3</th>
      <td>244</td>
      <td>51</td>
      <td>2</td>
      <td>880606923</td>
    </tr>
    <tr>
      <th>4</th>
      <td>166</td>
      <td>346</td>
      <td>1</td>
      <td>886397596</td>
    </tr>
  </tbody>
</table>
</div>



As you can see, this dataset could easily be represented as a bimodal weighted network graph connecting user nodes with movies nodes with rating weights. Let's also import some metadata concerning the movies to bring the scenario to life.


```python
col_names = ['movie_id' , 'movie_title' , 'release_date' , 'video_release_date' ,
             'IMDb_URL' , 'unknown', 'Action', 'Adventure', 'Animation',
             'Childrens', 'Comedy', 'Crime' , 'Documentary', 'Drama', 'Fantasy',
             'Film-Noir', 'Horror', 'Musical', 'Mystery' , 'Romance' , 'Sci-Fi',
             'Thriller', 'War' ,'Western']
movies = pd.read_csv('ml-100k/u.item', delimiter='|', encoding='latin1', names=col_names)
movies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>movie_title</th>
      <th>release_date</th>
      <th>video_release_date</th>
      <th>IMDb_URL</th>
      <th>unknown</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Childrens</th>
      <th>...</th>
      <th>Fantasy</th>
      <th>Film-Noir</th>
      <th>Horror</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>01-Jan-1995</td>
      <td>NaN</td>
      <td>http://us.imdb.com/M/title-exact?Toy%20Story%2...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>GoldenEye (1995)</td>
      <td>01-Jan-1995</td>
      <td>NaN</td>
      <td>http://us.imdb.com/M/title-exact?GoldenEye%20(...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Four Rooms (1995)</td>
      <td>01-Jan-1995</td>
      <td>NaN</td>
      <td>http://us.imdb.com/M/title-exact?Four%20Rooms%...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Get Shorty (1995)</td>
      <td>01-Jan-1995</td>
      <td>NaN</td>
      <td>http://us.imdb.com/M/title-exact?Get%20Shorty%...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Copycat (1995)</td>
      <td>01-Jan-1995</td>
      <td>NaN</td>
      <td>http://us.imdb.com/M/title-exact?Copycat%20(1995)</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



## Transforming the Data


```python
user_ratings = df.pivot(index='user_id', columns='item_id', values='rating')
user_ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>item_id</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>1673</th>
      <th>1674</th>
      <th>1675</th>
      <th>1676</th>
      <th>1677</th>
      <th>1678</th>
      <th>1679</th>
      <th>1680</th>
      <th>1681</th>
      <th>1682</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1682 columns</p>
</div>



## Filling Missing Values


```python
for col in user_ratings:
    mean = user_ratings[col].mean()
    user_ratings[col] = user_ratings[col].fillna(value=mean)
user_ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>item_id</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>1673</th>
      <th>1674</th>
      <th>1675</th>
      <th>1676</th>
      <th>1677</th>
      <th>1678</th>
      <th>1679</th>
      <th>1680</th>
      <th>1681</th>
      <th>1682</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.000000</td>
      <td>3.206107</td>
      <td>3.033333</td>
      <td>3.550239</td>
      <td>3.302326</td>
      <td>3.576923</td>
      <td>3.798469</td>
      <td>3.995434</td>
      <td>3.896321</td>
      <td>2.000000</td>
      <td>...</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.878319</td>
      <td>3.206107</td>
      <td>3.033333</td>
      <td>3.550239</td>
      <td>3.302326</td>
      <td>3.576923</td>
      <td>3.798469</td>
      <td>3.995434</td>
      <td>3.896321</td>
      <td>3.831461</td>
      <td>...</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.878319</td>
      <td>3.206107</td>
      <td>3.033333</td>
      <td>3.550239</td>
      <td>3.302326</td>
      <td>3.576923</td>
      <td>3.798469</td>
      <td>3.995434</td>
      <td>3.896321</td>
      <td>3.831461</td>
      <td>...</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>3.033333</td>
      <td>3.550239</td>
      <td>3.302326</td>
      <td>3.576923</td>
      <td>3.798469</td>
      <td>3.995434</td>
      <td>3.896321</td>
      <td>3.831461</td>
      <td>...</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1682 columns</p>
</div>



## Creating a User Matrix

To create a user matrix, you must calculate the distance between users. Choosing an appropriate distance metric for this is crucial. In this instance, a simple Euclidean distance is apt to be appropriate, but in other instances an alternative metric such as cosine distance might be a more sensible choice.


```python
import numpy as np
import datetime
```


```python
u1 = user_ratings.iloc[1]
u2 = user_ratings.iloc[2]
def distance(v1,v2):
    return np.sqrt(np.sum((v1-v2)**2))
distance(u1,u2)
```




    11.084572689977236




```python
# ⏰ Expect this cell to take several minutes to run
start = datetime.datetime.now()
user_matrix = []
for i, row in enumerate(user_ratings.index):
    u1 = user_ratings[row]
    # Matrix is symetric, so fill in values for previously examined users
    user_distances = [entry[i] for entry in user_matrix] 
    for j, row2 in enumerate(user_ratings.index[i:]):
        u2 = user_ratings[row2]
        d = distance(u1,u2)
        user_distances.append(d)
    user_matrix.append(user_distances)
user_similarities = pd.DataFrame(user_matrix)

end = datetime.datetime.now()
elapsed = end - start
print(elapsed)

user_similarities.head()
```

    0:02:12.766052





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>933</th>
      <th>934</th>
      <th>935</th>
      <th>936</th>
      <th>937</th>
      <th>938</th>
      <th>939</th>
      <th>940</th>
      <th>941</th>
      <th>942</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>29.936426</td>
      <td>34.042510</td>
      <td>25.599772</td>
      <td>27.165580</td>
      <td>22.301547</td>
      <td>26.215828</td>
      <td>23.496667</td>
      <td>25.937816</td>
      <td>21.335516</td>
      <td>...</td>
      <td>36.156616</td>
      <td>26.799824</td>
      <td>19.717999</td>
      <td>25.405054</td>
      <td>36.780720</td>
      <td>21.812402</td>
      <td>51.343159</td>
      <td>32.668768</td>
      <td>23.666899</td>
      <td>24.014478</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29.936426</td>
      <td>0.000000</td>
      <td>16.182447</td>
      <td>19.619520</td>
      <td>13.942961</td>
      <td>17.161477</td>
      <td>28.271802</td>
      <td>29.750381</td>
      <td>30.305192</td>
      <td>23.904303</td>
      <td>...</td>
      <td>16.059514</td>
      <td>11.520504</td>
      <td>25.495994</td>
      <td>14.214126</td>
      <td>15.803102</td>
      <td>17.058759</td>
      <td>28.922541</td>
      <td>13.417856</td>
      <td>14.396717</td>
      <td>14.214562</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34.042510</td>
      <td>16.182447</td>
      <td>0.000000</td>
      <td>24.390253</td>
      <td>16.425187</td>
      <td>20.838161</td>
      <td>32.394615</td>
      <td>35.050119</td>
      <td>33.991216</td>
      <td>28.574367</td>
      <td>...</td>
      <td>13.944501</td>
      <td>13.948331</td>
      <td>30.359617</td>
      <td>17.340413</td>
      <td>13.335128</td>
      <td>21.472178</td>
      <td>24.388253</td>
      <td>13.221221</td>
      <td>19.026807</td>
      <td>18.205507</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25.599772</td>
      <td>19.619520</td>
      <td>24.390253</td>
      <td>0.000000</td>
      <td>18.809007</td>
      <td>15.341923</td>
      <td>24.285722</td>
      <td>23.233123</td>
      <td>24.219603</td>
      <td>18.588349</td>
      <td>...</td>
      <td>24.992752</td>
      <td>16.263677</td>
      <td>18.954594</td>
      <td>16.038223</td>
      <td>25.407118</td>
      <td>14.828270</td>
      <td>39.984010</td>
      <td>22.005445</td>
      <td>14.904607</td>
      <td>15.217085</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27.165580</td>
      <td>13.942961</td>
      <td>16.425187</td>
      <td>18.809007</td>
      <td>0.000000</td>
      <td>13.840300</td>
      <td>25.698150</td>
      <td>27.076469</td>
      <td>26.955596</td>
      <td>20.865873</td>
      <td>...</td>
      <td>16.513384</td>
      <td>9.004673</td>
      <td>21.955017</td>
      <td>11.236040</td>
      <td>16.516795</td>
      <td>13.212617</td>
      <td>31.007449</td>
      <td>13.597272</td>
      <td>12.242182</td>
      <td>11.385938</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 943 columns</p>
</div>



## Calculating Recommendations

Now on to the recommendations! To do this, you'll select the top $n$ users who are similar to the user in question. From there, you'll then predict the current user's rating of a movie based on the average of the closest users ratings. Finally, you'll then sort these ratings from highest to lowest and remove movies that the current user has already rated and seen. 


```python
def recommend_movies(user, user_similarities, user_ratings, df, n_users=20, n_items=10):
    """n is the number of similar users who you wish to use to generate recommendations."""
    # User_Similarities Offset By 1 and Must Remove Current User
    top_n_similar_users = user_similarities[user-1].drop(user-1).sort_values().index[:n_users] 
    # Again, fixing the offset of user_ids
    top_n_similar_users = [i+1 for i in top_n_similar_users] 
    already_watched = set(df[df.user_id == 0].item_id.unique())
    unwatched = set(df.item_id.unique()) - already_watched
    projected_user_reviews = user_ratings[user_ratings.index.isin(top_n_similar_users)].mean()[list(unwatched)].sort_values(ascending=False)
    return projected_user_reviews[:n_items]
```


```python
recommend_movies(1, user_similarities, user_ratings, df)
```




    item_id
    1122    5.0
    814     5.0
    1500    5.0
    1536    5.0
    1653    5.0
    1599    5.0
    1467    5.0
    1189    5.0
    1201    5.0
    1293    5.0
    dtype: float64



## Summary

In this lesson you got a proper introduction to recommendation systems using collaborative filtering!
