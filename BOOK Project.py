#!/usr/bin/env python
# coding: utf-8

# # Book Recommenmdation System

# In a very general way, recommender systems are algorithms aimed at suggesting relevant items to users (items being movies to watch, text to read, products to buy, or anything else depending on industries).
# Recommender systems are really critical in some industries as they can generate a huge amount of income when they are efficient or also be a way to stand out significantly from competitors. The main objective is to create a book recommendation system for users.

# In[1]:


#Importinf Libraries
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
# This is to supress the warning messages (if any) generated in our code
import warnings
warnings.filterwarnings('ignore')


# # Data
# ● Users :
# Contains the users. Note that user IDs (User-ID) have been anonymized and map to integers. Demographic data is provided (Location, Age) if available. Otherwise, these fields contain NULL values.
# ● Books :
# Books are identified by their respective ISBN. Invalid ISBNs have already been removed from the dataset. Moreover, some content-based information is given (Book-Title, Book-Author, Year-Of-Publication, Publisher), obtained from Amazon Web Services. Note that in the case of several authors, only the first is provided. URLs linking to cover images are also given, appearing in three different flavors (Image-URL-S, Image-URL-M, Image-URL-L), i.e., small, medium, large. These URLs point to the Amazon website.
# ● Ratings :
# Contains the book rating information. Ratings (Book-Rating) are either explicit, expressed on a scale from 1-10 (higher values denoting higher appreciation), or implicit, expressed by 0.

# # Importing Dataset

# In[2]:


#loading the required datasets
books= pd.read_csv(r"C:\Users\sneha-pc\Downloads\doc,groups\Books.csv",encoding="latin-1")
ratings = pd.read_csv(r"C:\Users\sneha-pc\Downloads\doc,groups\Ratings.csv",encoding="latin-1")
users = pd.read_csv(r"C:\Users\sneha-pc\Downloads\doc,groups\Users.csv",encoding="latin-1")


# In[ ]:





# In[ ]:





# In[ ]:





# # 1) Users_dataset

# In[3]:


def missing_values(df):
    mis_val=df.isnull().sum()
    mis_val_percent=round(df.isnull().mean().mul(100),2)
    mz_table=pd.concat([mis_val,mis_val_percent],axis=1)
    mz_table=mz_table.rename(
    columns={df.index.name:'col_name',0:'Missing Values',1:'% of Total Values'})
    mz_table['Data_type']=df.dtypes
    mz_table=mz_table.sort_values('% of Total Values',ascending=False)
    return mz_table.reset_index()


# In[4]:


missing_values(users)


#     Age have about 39% of missing values.

# In[5]:


#Age distribution


# In[6]:


users.Age.hist(bins=[0,10,20,30,40,50,100])
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# The most active users are ampong those in their 20-40s of age

# In[7]:


#Outliers detection
sns.boxplot(y = 'Age', data= users)
plt.title('Find outlier data in Age column')


# In[8]:


print(sorted(users.Age.unique()))


# Ok we have Outlier data in Age

# In[9]:


#lets find our unique value in location column


# In[10]:


users.Location.unique()


# In[11]:


users.Location.nunique()


# 57339 unique values, its really hard to understand
# so lets create column Country

# In[12]:


for i in users :
    users['Country'] = users.Location.str.extract(r'\,+\s?(\w*\s?\w*)\"*$')


# In[13]:


users.Country.nunique()


# In[14]:


#drop location column
users.drop('Location',axis=1,inplace=True)


# In[15]:


users


# In[16]:


users.isnull().sum()


# In[17]:


users['Country'] = users['Country'].astype('str')


# In[18]:


a = list(users.Country.unique())
a =set(a)
a= list(a)
a = [x for x in a if x is not None]
a.sort()
print(a)


# Some data has Misspellings , Lets correct it.

# In[19]:


users['Country'].replace(['','01776','02458','19104','23232','30064','85021','87510','alachua','america','austria','autralia','cananda','geermany','italia','united kindgonm','united sates','united staes','united state','united states','us'],
                           ['other','usa','usa','usa','usa','usa','usa','usa','usa','usa','australia','australia','canada','germany','italy','united kingdom','usa','usa','usa','usa','usa'],inplace=True)


# In[20]:


plt.figure(figsize=(15,7))
sns.countplot(y='Country',data=users,order=pd.value_counts(users['Country']).iloc[:10].index)
plt.title('Count of users Country wise')


# Most number of users are from USA

# In[21]:


#Lets treat outliers in users age


# In[22]:


sns.distplot(users.Age)
plt.title('Age Distribution Plot')


# Age value's below 5 and above 100 do not make much sense for our book rating case...hence replacing these by NaNs

# In[23]:


#Outlier data became NaN
users.loc[(users.Age > 100) | (users.Age < 5), 'Age'] = np.nan


# In[24]:


users.isna().sum()


# Age has positive Skewness (right tail) so we can use median to fill Nan values, but for this we don't like to fill Nan value just for one range of age. To handle this we'll use country column to fill Nan.

# In[25]:


users['Age'] = users['Age'].fillna(users.groupby('Country')['Age'].transform('median'))


# In[26]:


users.isna().sum()


# Still we have 276 Nan values let's fill them with mean

# In[27]:


users['Age'].fillna(users.Age.mean(), inplace=True)


# In[28]:


users.isna().sum()


# # 2)Books_Dataset

# In[29]:


books


# In[30]:


books.head(2)


# In[31]:


#Top 10 Authors which have written the most books.


# In[32]:


plt.figure(figsize=(15,7))
sns.countplot(y='Book-Author',data=books,order=pd.value_counts(books['Book-Author']).iloc[:10].index)
plt.title('Top 10 Authors')


# In[33]:


#Top 10 Publisher which have published the most books


# In[34]:


plt.figure(figsize=(15,7))
sns.countplot(y='Publisher',data=books,order=pd.value_counts(books['Publisher']).iloc[:10].index)
plt.title('Top 10 Publishers')


# In[35]:


books['Year-Of-Publication']=books['Year-Of-Publication'].astype('str')
a=list(books['Year-Of-Publication'].unique())
a=set(a)
a=list(a)
a = [x for x in a if x is not None]
a.sort()
print(a)


# In[36]:


#investigating the rows having 'DK Publishing Inc' as year Of Publication
books.loc[books['Year-Of-Publication'] == 'DK Publishing Inc',:]


# As it can be seen from above that there are some incorrect entries in Year-Of-Publication field. It looks like Publisher names 'DK Publishing Inc' and 'Gallimard' have been incorrectly loaded as Year-Of-Publication in dataset due to some errors in csv file

# In[37]:


#From above, it is seen that bookAuthor is incorrectly loaded with bookTitle, hence making required corrections
#ISBN '0789466953'
books.loc[books.ISBN == '0789466953','Year-Of-Publication'] = 2000
books.loc[books.ISBN == '0789466953','Book-Author'] = "James Buckley"
books.loc[books.ISBN == '0789466953','Publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '0789466953','Book-Title'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"


# In[38]:


#ISBN '078946697X'
books.loc[books.ISBN == '078946697X','Year-Of-Publication'] = 2000
books.loc[books.ISBN == '078946697X','Book-Author'] = "Michael Teitelbaum"
books.loc[books.ISBN == '078946697X','Publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '078946697X','Book-Title'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"


# In[39]:


#rechecking
books.loc[(books.ISBN == '0789466953') | (books.ISBN == '078946697X'),:]


# In[40]:


#investigating the rows having 'Gallimard' as yearOfPublication
books.loc[books['Year-Of-Publication'] == 'Gallimard',:]


# In[41]:


#making required corrections as above, keeping other fields intact
books.loc[books.ISBN == '2070426769','Year-Of-Publication'] = 2003
books.loc[books.ISBN == '2070426769','Book-Author'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
books.loc[books.ISBN == '2070426769','Publisher'] = "Gallimard"
books.loc[books.ISBN == '2070426769','Book-Title'] = "Peuple du ciel, suivi de 'Les Bergers"


books.loc[books.ISBN == '2070426769',:]


# In[42]:


books['Year-Of-Publication']=pd.to_numeric(books['Year-Of-Publication'], errors='coerce')

print(sorted(books['Year-Of-Publication'].unique()))
#Now it can be seen that yearOfPublication has all values as integers


# The value 0 for Year-Of_Publication is invalid and as this dataset was published in 2004, We have assumed that the years after 2006 to be invalid and setting invalid years as NaN
# 
# Reference of the fact: http://www2.informatik.uni-freiburg.de/~cziegler/BX/

# In[43]:


books.loc[(books['Year-Of-Publication'] > 2006) | (books['Year-Of-Publication'] == 0),'Year-Of-Publication'] = np.NAN

#replacing NaNs with median value of Year-Of-Publication
books['Year-Of-Publication'].fillna(round(books['Year-Of-Publication'].median()), inplace=True)


# In[44]:


#dropping last three columns containing image URLs which will not be required for analysis
books.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'],axis=1,inplace=True)


# In[45]:


books.isna().sum()


# In[46]:


#exploring 'publisher' column
books.loc[books.Publisher.isnull(),:]


# In[47]:


#Filling Nan of Publisher with others
books.Publisher.fillna('other',inplace=True)


# In[48]:


#exploring 'Book-Author' column
books.loc[books['Book-Author'].isnull(),:]


# In[49]:


#Filling Nan of Book-Author with others
books['Book-Author'].fillna('other',inplace=True)


# In[50]:


books.isna().sum()


# # 3) Ratings Dataset

# In[51]:


ratings


# Ratings dataset should have books only which exist in our books dataset

# In[52]:


ratings_new = ratings[ratings.ISBN.isin(books.ISBN)]
ratings.shape, ratings_new.shape


# It can be seen that many rows having book ISBN not part of books dataset got dropped off
# 
# Ratings dataset should have ratings from users which exist in users dataset.
# 
# 

# In[53]:


print("Shape of dataset before dropping",ratings_new.shape)
ratings_new = ratings_new[ratings_new['User-ID'].isin(users['User-ID'])]
print("shape of dataset after dropping",ratings_new.shape)


# It can be seen that no new user was there in ratings dataset.
# 
# Let's see how the ratings are distributed

# In[54]:


plt.rc("font", size=15)
ratings_new['Book-Rating'].value_counts(sort=False).plot(kind='bar')
plt.title('Rating Distribution\n')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()


# The ratings are very unevenly distributed, and the vast majority of ratings are 0 .As quoted in the description of the dataset - BX-Book-Ratings contains the book rating information. Ratings are either explicit, expressed on a scale from 1-10 higher values denoting higher appreciation, or implicit, expressed by 0.Hence segragating implicit and explict ratings datasets

# In[55]:


#Hence segragating implicit and explict ratings datasets
ratings_explicit = ratings_new[ratings_new['Book-Rating'] != 0]
ratings_implicit = ratings_new[ratings_new['Book-Rating'] == 0]


# In[56]:


print('ratings_explicit dataset',ratings_explicit.shape)
print('ratings_implicit dataset',ratings_implicit.shape)


# In[57]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(12, 8))
sns.countplot(data=ratings_explicit , x='Book-Rating', palette='rocket_r')


# It can be observe that higher ratings are more common amongst users and rating 8 has been rated highest number of times

# In[58]:


#Let's find the top 5 books which are rated by most number of users.
rating_count = pd.DataFrame(ratings_explicit.groupby('ISBN')['Book-Rating'].count())
rating_count.sort_values('Book-Rating', ascending=False).head()


# The book with ISBN '0316666343' received the most rating counts.

# In[59]:


#Let’s find out what book it is, and what books are in the top 5.
most_rated_books = pd.DataFrame(['0316666343', '0971880107', '0385504209', '0312195516', '0060928336'], index=np.arange(5), columns = ['ISBN'])
most_rated_books_summary = pd.merge(most_rated_books, books, on='ISBN')
most_rated_books_summary


# The book that received the most rating counts in this data set is Rich Shapero’s “Wild Animus”. And there is something in common among these five books that received the most rating counts — they are all novels. So it is conclusive that novels are popular and likely receive more ratings.

# In[60]:


#create column Rating Average
ratings_explicit ['Avg_Rating'] = ratings_explicit.groupby('ISBN')['Book-Rating'].transform('mean')


# In[61]:


#Cearting column Rating sum
ratings_explicit['Total_No_Of_Users_Rated']=ratings_explicit.groupby('ISBN')['Book-Rating'].transform('count')


# In[62]:


ratings_explicit


# # Merging All Dataset

# In[63]:


Final_Dataset=users.copy()
Final_Dataset=pd.merge(Final_Dataset,ratings_explicit,on='User-ID')
Final_Dataset=pd.merge(Final_Dataset,books,on='ISBN')


# In[64]:


Final_Dataset


# In[65]:


missing_values(Final_Dataset)


# In[66]:


Final_Dataset.shape


# # Collaborative Filtering based Recommendation System

# In[67]:


Final_Dataset


# In[68]:


x = Final_Dataset.groupby('User-ID').count()['Book-Rating'] >50
y = x[x].index


# In[69]:


x


# In[70]:


y


# In[71]:


filtered_rating = Final_Dataset[Final_Dataset['User-ID'].isin(y)]


# In[72]:


filtered_rating


# In[73]:


y1 = filtered_rating.groupby('Book-Title').count()['Book-Rating']>= 10
famous_books = y1[y1].index


# In[74]:


famous_books


# In[75]:


final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books) ]
final_ratings


# In[76]:


final_ratings['User-ID'].nunique()


# # Item- Item Based Filtering

# In[77]:


pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')


# In[78]:


pt


# In[79]:


pt.fillna(0,inplace=True)


# In[80]:


pt


# In[81]:


#Calculate the distances and similarity


# In[82]:


from sklearn.metrics import pairwise_distances


# In[83]:


pd.DataFrame(pairwise_distances(pt, metric='cosine'))


# In[84]:


sim = 1- pairwise_distances(pt, metric='cosine')
pd.DataFrame(sim)


# In[85]:


np.fill_diagonal(sim,0)


# In[86]:


similarity_item = pd.DataFrame(sim)
similarity_item


# In[87]:


similarity_item.index = final_ratings['Book-Title'].unique()
similarity_item.columns = final_ratings['Book-Title'].unique()


# In[88]:


pd.set_option('display.max_columns', None)


# In[89]:


similarity_item


# In[90]:


#Find out similar users


# In[91]:


similarity_item.idxmax()


# In[92]:


from sklearn.metrics.pairwise import cosine_similarity


# In[93]:


similarity_item_scores=cosine_similarity(pt)


# In[94]:


similarity_item_scores[0]


# In[95]:


similarity_item_scores.shape


# In[96]:


import pickle
pickle.dump(similarity_item_scores,open('similarity_item_score.pkl','wb'))


# In[97]:


def recommend_for_item(book_name):
    index=np.where(pt.index==book_name)[0][0]
    distances=similarity_item_scores[index]
    similar_items=sorted(list(enumerate(similarity_item_scores[index])),key=lambda x:x[1], reverse=True)[1:11]
    data=[]
    for i in similar_items:
        item=[]
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))

        data.append(item)
    return data


# In[98]:


recommend_for_item('The Notebook')


# # User-User Based Filtering

# In[99]:


final_ratings


# In[100]:


pt2 = final_ratings.pivot_table(index='User-ID',columns='Book-Title',values='Book-Rating')


# In[101]:


pt2


# In[102]:


pt2.fillna(0,inplace=True)


# In[103]:


pt2


# In[104]:


#Calculating distance and Similarity


# In[105]:


pd.DataFrame(pairwise_distances(pt2, metric='cosine'))


# In[106]:


sim2 = 1- pairwise_distances(pt2, metric='cosine')
pd.DataFrame(sim2)


# In[107]:


np.fill_diagonal(sim2,0)


# In[108]:


similarity_user = pd.DataFrame(sim2)


# In[109]:


similarity_user.index = final_ratings['User-ID'].unique()
similarity_user.columns = final_ratings['User-ID'].unique()


# In[110]:


similarity_user


# In[111]:


#Finind similar user


# In[112]:


similarity_user.idxmax()


# In[113]:


final_ratings[(final_ratings['User-ID'] == 72352) | (final_ratings['User-ID'] == 132492)]


# In[114]:


similarity_user_score = cosine_similarity(pt2)


# In[115]:


similarity_user_score[0]


# In[116]:


def recommendations_for_user(user_id):
    print('\n Recommended Books for User_id',(user_id),':\n')
    recom = list(similarity_user.sort_values([user_id], ascending= False).head().index)[1:11]
    books_list = []
    for i in recom:
        books_list = books_list + list(final_ratings[final_ratings['User-ID']==i]['Book-Title'])
    return set(books_list)-set(final_ratings[final_ratings['User-ID']==user_id]['Book-Title'])


# In[117]:


def recommendations_for_user(user_id):
    print('\nRecommended Books for User_id', user_id, ':\n')
    if user_id in similarity_user.columns:
        recom = list(similarity_user.sort_values(user_id, ascending=False).head().index)[1:11]
        books_list = []
        for i in recom:
            books_list += list(final_ratings[final_ratings['User-ID'] == i]['Book-Title'])
        recommended_books = set(books_list) - set(final_ratings[final_ratings['User-ID'] == user_id]['Book-Title'])
        return list(recommended_books)[:10]  # Limit the recommendations to the top 10
    else:
        return []


# In[118]:


recommendations_for_user(6242)


# In[ ]:





# # Model Evaluation

# In[119]:


#Train-Test Split
from sklearn import model_selection
train_data, test_data = model_selection.train_test_split(final_ratings, test_size=0.20)


# In[120]:


print(f'Training set lengths: {len(train_data)}')
print(f'Testing set lengths: {len(test_data)}')
print(f'Test set is {(len(test_data)/(len(train_data)+len(test_data))*100):.0f}% of the full dataset.')


# In[121]:


# Get int mapping for user_id in train dataset
u_unique_train = train_data['User-ID'].unique()
train_data_user2idx = {o:i for i, o in enumerate(u_unique_train)}

# Get int mapping for isbn in train dataset
i_unique_train = train_data['ISBN'].unique()
train_data_book2idx = {o:i for i, o in enumerate(i_unique_train)}


# In[122]:


# Get int mapping for user_id in test dataset
u_unique_test = test_data['User-ID'].unique()
test_data_user2idx = {o:i for i, o in enumerate(u_unique_test)}

# Get int mapping for isbn in test dataset
i_unique_test = test_data['ISBN'].unique()
test_data_book2idx = {o:i for i, o in enumerate(i_unique_test)}


# In[123]:


# training set
train_data['u_unique'] = train_data['User-ID'].map(train_data_user2idx)
train_data['i_unique'] = train_data['ISBN'].map(train_data_book2idx)

# testing set
test_data['u_unique'] = test_data['User-ID'].map(test_data_user2idx)
test_data['i_unique'] = test_data['ISBN'].map(test_data_book2idx)

# Convert back to three feature of dataframe
train_data = train_data[['u_unique', 'i_unique', 'Book-Rating']]
test_data = test_data[['u_unique', 'i_unique', 'Book-Rating']]


# In[124]:


train_data.sample(2)


# In[125]:


test_data.sample(2)


# In[126]:


# User-Item for Train Data


# In[127]:


# first I'll create an empty matrix of users books and then I'll add the appropriate values to the matrix by extracting them from the dataset
n_users = train_data['u_unique'].nunique()
n_books = train_data['i_unique'].nunique()

train_matrix = np.zeros((n_users, n_books))

for entry in train_data.itertuples():
    train_matrix[entry[1]-1, entry[2]-1] = entry[3]
    # entry[1] is the user-id, entry[2] is the book-isbn and -1 is to counter 0-based indexing


# In[128]:


train_matrix.shape


# In[129]:


# User-Item for Test Data


# In[130]:


n_users = test_data['u_unique'].nunique()
n_books = test_data['i_unique'].nunique()

test_matrix = np.zeros((n_users, n_books))

for entry in test_data.itertuples():
    test_matrix[entry[1]-1, entry[2]-1] = entry[3]


# In[131]:


test_matrix.shape


# # Cosine Similarity Based Recommendation System

# In[132]:


train_matrix_small = train_matrix[:1000, :1000]
test_matrix_small = test_matrix[:1000, :1000]

from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_matrix_small, metric='cosine')
item_similarity = pairwise_distances(train_matrix_small.T, metric='cosine')


# In[133]:


# function to predict the similarity :
def predict_books(ratings, similarity, type='user'): # default type is 'user'
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)

        # Use np.newaxis so that mean_user_rating has the same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


# In[134]:


item_prediction = predict_books(train_matrix_small, item_similarity , type='item')
user_prediction = predict_books(train_matrix_small, user_similarity , type='user')


# # Evaluation Metric

# In[135]:


# Evaluation metric by mean squared error
from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(prediction, test_matrix):
    prediction = prediction[test_matrix.nonzero()].flatten()
    test_matrix = test_matrix[test_matrix.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, test_matrix))

print(f'Item-based CF RMSE: {rmse(item_prediction, test_matrix_small)}')
print(f'User-based CF RMSE: {rmse(user_prediction, test_matrix_small)}')


# In[136]:


## Let's go through Model based approach by SVD model.


# In[137]:


from surprise import Reader, Dataset


# In[138]:


# Creating a 'Reader' object to set the limit of the ratings
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(final_ratings[['User-ID','Book-Title','Book-Rating']], reader)


# In[139]:


from surprise import SVD, model_selection, accuracy
model = SVD()

# Train on books dataset
get_ipython().run_line_magic('time', "model_selection.cross_validate(model, data, measures=['RMSE'], cv=5, verbose=True)")


# # Train - Test Split

# In[140]:


# train and test split
trainset, testset = model_selection.train_test_split(data, test_size=0.2)

# SVD model
model = SVD()
model.fit(trainset)


# # Evaluation metrics for SVD model

# In[141]:


# displaying RMSE score
predictions = model.test(testset)
print(f"The accuracy is {accuracy.rmse(predictions)}")


# # Testing Results

# In[142]:


# to test result let's take an user-id and item-id to test our model.
uid = 276744
iid = '038550120X'
pred = model.predict(uid, iid, verbose=True)


# In[143]:


# display estimated rating and real rating
print(f'The estimated rating for the book with ISBN code {pred.iid} from user #{pred.uid} is {pred.est:.2f}.\n')
actual_rtg= ratings_explicit[(ratings_explicit['User-ID']==pred.uid) &
                             (ratings_explicit['ISBN']==pred.iid)]['Book-Rating'].values[0]
print(f'The real rating given for this was {actual_rtg:.2f}.')


# In[146]:


import pickle
pickle.dump(similarity_user,open('similarity_user.pkl','wb'))

