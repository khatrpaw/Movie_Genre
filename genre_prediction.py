#######################################################
#Movie Genre Predicition
#######################################################
#Import libraries

import sys
import pandas as pd
import numpy as np
import nltk
import re
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression


# Binary Relevance
from sklearn.multiclass import OneVsRestClassifier

# Performance metric
from sklearn.metrics import f1_score
#################################################
# Function to extract year from the movie name
# For eg : Toy Story (1995) -> Toy Story
################################################# 

def extractYear(movie_name):
	i = re.search("\(",movie_name)
	records = movie_name[:i.start()-1]
	return records

#################################################
# Function for text cleaning by removing backslash-apostrophe,everything except alphabets,whitespaces
# and converting into lowercase  
################################################# 
 
def clean_text(text):
    #print text
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text) 
    # remove everything except alphabets 
    text = re.sub("[^a-zA-Z]"," ",text) 
    # remove whitespaces 
    text = ' '.join(text.split()) 
    # convert text to lowercase 
    text = text.lower() 
    return text
#################################################
# Function to find most frequent words  
#################################################
 	
def freq_words(x, terms = 20): 
  all_words = ' '.join([text for text in x]) 
  all_words = all_words.split() 
  fdist = nltk.FreqDist(all_words) 
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())}) 
  
  # selecting top 20 most frequent words 
  d = words_df.nlargest(columns="count", n = terms) 
  
  # visualize words and frequencies
  plt.figure(figsize=(12,15)) 
  ax = sns.barplot(data=d, x= "count", y = "word") 
  ax.set(ylabel = 'Word') 
  plt.show()

  
pd.set_option('display.max_colwidth', 100) #to see the data in the terminals

#Reading the csv files 
meta_movie = pd.read_csv("movies_metadata.csv",low_memory = False) #To get the movie name
meta_genre = pd.read_csv("movies.csv",low_memory = False) #To get the genre of the movie

print meta_movie.head()
print meta_genre.head()

#Remove year from the movie_name of movies.csv data
movie_name = []
for row in tqdm(meta_genre['title']):
	   name = extractYear(row)	
	   movie_name.append(name)	
#Converting the genre
           
# create dataframe
movies = pd.DataFrame({'movie_name': meta_movie['original_title'], 'description': meta_movie['overview']})
genre = pd.DataFrame({'movie_name' : movie_name,'genre':meta_genre['genres']})


print movies.head()
print genre.head()

movies = pd.merge(movies,genre[['movie_name','genre']],on = 'movie_name')
print movies.head()

# an empty list
genres = [] 

#If the movie description is NA then replace to "No description"
i=0
for row in tqdm(movies['description']): 
	if(pd.isnull(row)):
		movies['description'][i] = "No description" 		
	i = i+1	

# Split genres and convert into dict
for record in tdm(movies['genre']):	 
	record = record.split('|')
	genres.append(record)
	
# add to 'movies' a new col genre_new  
movies['genre_new'] = genres
print movies.head()


# remove samples with 0 genre tags
movies_new = movies[~(movies['genre_new'].str.len() == 0)]
#print movies_new.shape, movies.shape

# get all genre tags in a list
all_genres = sum(genres,[])
#print len(set(all_genres))
###################################################################################################################
#FreqDist function from the nltk library to create a dictionary of genres and their occurrence count across the dataset
#To find the count the movies in each genre across the dataset 

#all_genres = nltk.FreqDist(all_genres) 
#all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()),'Count': list(all_genres.values())})

#g = all_genres_df.nlargest(columns="Count", n = 50) 
#plt.figure(figsize=(12,15)) 
#ax = sns.barplot(data=g, x= "Count", y = "Genre") 
#ax.set(ylabel = 'Count') 
#plt.show()
######################################################################################################################
		
#Cleaning Description by calling clean_text		
movies_new['clean_description'] = movies_new['description'].apply(lambda x: clean_text(x))

#Trying most frequency of words in description
#freq_words(movies_new['description'], 100)

#Downloading english stopwords like (this,that,at,of etc) to get keywords to predect   
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

#Removing stopwords and keeping description with keywords
movies_new['clean_description'] = movies_new['clean_description'].apply(lambda x: remove_stopwords(x))

#Frequency of the words
#freq_words(movies_new['clean_description'], 100)

#Converting Label text to features 
multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(movies_new['genre_new'])

# transform target variable by taking most frequent 600 words 
y = multilabel_binarizer.transform(movies_new['genre_new'])
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=600)

# split dataset into training and validation set spliting me 70% - 30%
xtrain, xval, ytrain, yval = train_test_split(movies_new['clean_description'], y, test_size=0.3, random_state=9)

# create TF-IDF features
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)

lr = LogisticRegression()
clf = OneVsRestClassifier(lr)


# fit model on train data
clf.fit(xtrain_tfidf, ytrain)
# make predictions for validation set
y_pred = clf.predict(xval_tfidf)


#print y_pred[3]
#print multilabel_binarizer.inverse_transform(y_pred)[3]

# evaluate performance
print f1_score(yval, y_pred, average="micro")

# predict probabilities
y_pred_prob = clf.predict_proba(xval_tfidf)

t = 0.5 # threshold value
y_pred_new = (y_pred_prob >= t).astype(int)

# evaluate performance
print f1_score(yval, y_pred_new, average="micro")


def infer_tags(q):
    q = clean_text(q)
    q = remove_stopwords(q)
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = clf.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)

#Showing Output if you want to check from the first val 5 test  
for i in range(10): 
  k = xval.sample(1).index[0]	 
  print("Movie: ", movies_new['movie_name'][k]) 
  print("Description: ",movies_new['clean_description'][k]) 
  print("Predicted genre: ", infer_tags(xval[k]))
  print("Actual genre: ",movies_new['genre_new'][k])
  print(" ")


