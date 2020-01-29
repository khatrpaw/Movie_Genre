# Movie_Genre


Predicting Genre of the movie

The dataset which conatins movie_metadata.csv and movies.csv is used downloaded from [Movies Lens ](
 https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7#movies_metadata.csv) [need to be downlaoded cannot attach due to size] and [IMDB] (https://drive.google.com/file/d/1Dn1BZD3YxgBQJSIjbfNnmCFlDW2jdQGD/view) respectively.
 
 movie-metadata.csv conatins orignial title and description and movie genre conatins movie name and genre.Both have the same movie names.
 
Library

    *  Natural Language Toolkit - import nltk - to remove stopwords which are in English
    *  Pandas
    *  Numpy
    *  MatplotLib
    *  Skiit-learn      - for model creating,feature extraction 
    
 Alogrithm --
 
   1.Extract the data
   2.Data Exploration and Preprocessing
      1. Removing special character
      2. Clean the data
      3. Finding the keyword 
   3. Extracting Feature - TF-IDF indicates what the importance of the word is in order to understand the document or dataset. 
   4.Training the model
    
It take 10 samples from test data.    
Run --  python genre_prediction.py
for eg: 
Command-line :

python genre_prediction.py

Output -- 

('Movie: ', 'Hamlet')

('Description: ', "Tony Richardson's Hamlet is based on his own stage production. Filmed entirely within the Roundhouse in London (a disused train shed), it is shot almost entirely in close up, focusing the attention on faces and language rather than action.")

('Predicted genre: ', [('Drama',)])

('Actual genre: ', ['Drama'])


('Movie: ', 'Murder by Numbers')

('Description: ', 'Tenacious homicide detective Cassie Mayweather and her still-green partner are working a murder case, attempting to profile two malevolently brilliant young men: cold, calculating killers whose dark secrets might explain their crimes.')

('Predicted genre: ', [('Crime', 'Drama')])

('Actual genre: ', ['Crime', 'Thriller'])
   
..
