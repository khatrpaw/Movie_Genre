# Movie_Genre


Predicting Genre of the movie

The dataset which conatins movie_metadata.csv and movies.csv is used downloaded from [Movies Lens ](
 https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7#movies_metadata.csv) [need to be downlaoded cannot attach due to size] and [IMDB] (https://drive.google.com/file/d/1Dn1BZD3YxgBQJSIjbfNnmCFlDW2jdQGD/view) respectively.
 
 
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

Output -- ('Movie: ', 'Darkness Falls')

('Description: ', 'vengeful spirit taken form tooth fairy exact vengeance town lynched years earlier opposition child grown survived')

('Predicted genre: ', [('Drama',)])

('Actual genre: ', ['Horror', 'Thriller'])

    
