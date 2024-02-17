<h1 align="center">  Project_01-Book-Recommendation-System
 </h1>



<p align="center"> 
<img src="https://assets.website-files.com/6141c89a3874c3702674a1c0/62b1a359a17ffc3c6aaf2d3a_memgraph-building-real-time-book-recommendations-for-bookworms-release-blog-cover.png" height="400px">
</p>


<p> </p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :floppy_disk: Table of Content</h2>

 
  * [Introduction](#Introduction)
  * [Problem Statement](#Problem-Statement)
  * [Dataset Information](#dataset-information)
  * [Tools and Technologies used](#tools-and-technologies-used)
  * [Steps involved](#Steps-involved)
  * [Approaches used](#Approaches-used)
  * [Conclusion](#Conclusion)


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)


<h2> 📄 Introduction</h2>

With an increasing amount of information on the internet and a considerable increase in the number of users, it is essential for companies to search, map, and offer relevant information based on the preferences of users. Aan important functional means of providing personalized service to users is *Recommendation System*. This system uses algorithms and data analysis techniques to suggest items,content, or services that should be of interest to customers based on their past choices or by analyzing the preferences of similar users. Companies like Netflix, Amazon, etc. use recommender systems to help their users to identify the correct product or content for them. 

🎯 The main objective of this project  is to create a Book recommendation system that best predicts user interests and recommend the suitable/appropriate books to them ,using various approaches.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)


<h2> ❓ Problem Statement</h2>

 In some industries, the use of recommender systems is crucial because, when implemented well, they can be extremely profitable and set themselves apart from their competitors. Online book selling websites nowadays are competing with each other by many means.One of the most effective strategies for increasing sales,enhancing customer experience and retaining customers is building an efficient Recommendation system. The book recommendation system must recommend books that are of interest to buyers. Popularity based approach and Collaborative filtering approach are used in this project to build book recommendation systems.



![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)


<h2> :book: Dataset information </h2>

 Dataset used in this project is the Amazon Book-crossing dataset.This dataset has been compiled by Cai-Nicolas Ziegler in 2004, and it comprises of three file.They are: 

*Users*

* User-ID: A unique identification number for each user
* Location:It contains city,state and country  to which the user belongs ,separated by commas
* Age:The age of the user

*Books*

* ISBN:International Standard Book Number unique to each edition of the book
* Book-Title:Title of the book
* Book-Author:Author of the book(incase of several authors only the first is provided)
* Year-of-Publication:The year in which the particular edition of the book was published
* Publisher:Name of the Book Publishing company
* Image-URL-S: URL link to a small version of the book cover displayed on the Amazon website
* Image-URL-M:	URL link to Medium version image of the book cover displayed on the Amazon website
* Image-URL-L: URL link to Large sized image of the book cover displayed on the Amazon website

*Ratings*

* User-ID:as mentioned above
* ISBN:as mentioned above
* Book-Rating: The rating given by the user (identified by User-ID) for the book (identified by ISBN). It is either explicit,expressed on a scale from 1-10 (higher values denoting higher appreciation), or implicit,expressed by 0.



![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2>🛠 Tools and Technologies used </h2>


The programming language used in this project is Python . The following libraries were used for data analysis and data visualization and to build a recommender
model to recommend books as per user ID.
* *Pandas* :  For loading the dataset and performing data wrangling
* *Matplotlib*: For  data visualization.
* *Seaborn*: For data visualization.
* *NumPy*: For some math operations in predictions.
* *Statsmodels*: For statistical computations
* *Sklearn*:  For the purpose of analysis,prediction and evaluation.
* *Streamlit*: For the purpose of deployment of the model to get recommendation as per user ID.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> 📑 Steps involved </h2>

* *Data Preprocessing* : Checked for outliers, incorrect values, missing values, duplicate, performed data type correction and string formatting.
* *Merging of datasets* : In this project, recommender systems were built utilizing only explicit ratings . So finally,a new dataframe by merging the books dataset ,explicit ratings dataset and users dataset.
* *Feature Extraction* : Created new columns such as age_group by binning the 'Age' column and extracted the country name from the 'Location' column  .
* *Exploratory Data Analysis* : Performed Univariate, Bivariate, and Multivariate analysis with various graphs and plots to better understand the distribution of features and their relationships.
* *Implementation of various Recommender System approaches* :This project explores various recommendation algorithms, including Popularity Based Filtering, Collaborative Filtering based Recommendation System–(User-User based), and Collaborative Filtering based Recommendation System–(Item-Item Based).
* *Evaluation metrics* : Model evaluation metrics is important to distinguish the best collaborative filtering – either by memory based or model based approach.
* *Deployment of the model* : After model building and model evaluation next import key point is deployment. We have done deployment of recommender system by using streamlit tool and spyder app.
Following links are used to shows us the interface of the model and get recommendation of books as per user –id.
 Local URL: http://localhost:8501
 Network URL: http://192.168.208.127:8501


  

 



![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)


<h2>💻 Approaches used</h2>

The approaches used in this project are:

1. Popularity Based recommendation system
 
 It is a type of recommendation system that bases choices on factors like popularity and/or current trends. These systems determine which item (in this case,books) are in the trending list or are the most well-liked by users and then directly recommend them.
 
   - Country-wise approach
   - Author-wise approach
 
2. Collaborative Filitering Based recommendation system
 
 The Collaborative Filtering approach first investigates the user’s behaviors, interests, and searches for similar users.  It recommends items to users based on the ratings of similar users on various items and by predicting the missing ratings of the items . CF is broadly classified as memory-based and model-based CF.


   - Memory Based approach - KNN (Cosine similarity between items)
   - Model Based approach- SVD based recommendation system (prediction of ratings)
     - User-User based
     - Item-Item based

<h2> :bulb: Conclusion</h2>

* Starting with loading the data so far we have done data cleaning and feature engineering, null values treatment, some univariate analysis. Collaborative Filtering was among best method to approach recommendation system for this project. Model based approach like Latent Factor Model called SVD and Memory based approach with cosine similarity was model building approach. The comparison of RMSE score between model and memory based approach was quite different. Model Evaluation metrics shows better recommendation with model based CF. The RMSE score varies in both the model but optimal model we can find is in SVD. 
* Model evaluation metrics is important to distinguish the best collaborative filtering – either by memory based or model based approach. The memory based approach – Cosine Similarity shows RMSE score for item based CF is 8.07 and for user based CF it shows 8.06. The score is slightly similar. Model based collaborative filtering made it better score with Latent Factor Model called SVD. The score improved to 1.49 for both SVD RMSE and accuracy score.
* SVD with RMSE score is the best model with 1.49 for this dataset. This performance could be due to various reasons : pattern of data, different model give different accuracy score, business understanding, machine learning approach etc. Finally Singular Value Decomposition (SVD) is an optimal model for book recommendation system of this dataset.



