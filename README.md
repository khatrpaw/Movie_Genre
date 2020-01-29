# Movie_Genre


Predicting Genre of the movie

The dataset which conatins movie_metadata.csv and movies.csv is used downloaded from [Movies Lens ](
 https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7#movies_metadata.csv)
 
 
Library

    *  Natural Language Toolkit - import nltk - to remove stopwords which are in English
    *  Pandas
    *  Numpy
    *  MatplotLib
    *  Skiit-learn      - for model creating,feature extraction 
    
Run --  python genre_prediction.py

Output -- ('Movie: ', 'Darkness Falls')
('Description: ', 'vengeful spirit taken form tooth fairy exact vengeance town lynched years earlier opposition child grown survived')
('Predicted genre: ', [('Drama',)])
('Actual genre: ', ['Horror', 'Thriller'])

    
