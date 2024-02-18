import pandas as pd
import numpy as np
from re import match
credits_df = pd.read_excel("credits.xlsx")
movies_df = pd.read_excel("movies.xlsx")
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)

movies_df=movies_df.merge(credits_df, on="title")

movies_df=movies_df[['id','title','overview','genres','keywords','cast','crew']]

movies_df.isnull().sum()

movies_df.dropna(inplace=True)

import ast

def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies_df['genres']=movies_df['genres'].apply(convert)
movies_df['keywords']=movies_df['keywords'].apply(convert)

def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

movies_df['cast']=movies_df['cast'].apply(convert3)


def fetch(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L

movies_df['crew']=movies_df['crew'].apply(fetch)

movies_df['overview']=movies_df['overview'].apply(lambda x:x.split())

movies_df['genres']=movies_df['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['keywords']=movies_df['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['cast']=movies_df['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['crew']=movies_df['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies_df['tags']=movies_df['overview']+movies_df['genres']+movies_df['keywords']+movies_df['cast']+movies_df['crew']

new_df=movies_df[['id','title','tags']]

new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: ' '.join(x))
new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: x.lower())

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features= 5000, stop_words='english')

vectors=cv.fit_transform(new_df['tags']).toarray()

import nltk

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df.loc[:, 'tags'] = new_df['tags'].apply(stem)

from sklearn.metrics.pairwise import cosine_similarity

similarity=cosine_similarity(vectors)

def recommend(movie):
    movie_lower = movie.lower()
    movie_index = new_df[new_df['title'].apply(lambda x: str(x).lower()) == movie_lower].index
    if len(movie_index) > 0:
        movie_index = movie_index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        for i in movies_list:
            print(new_df.iloc[i[0]].title)
    else:
        print(f"\nsorry :( i have not watched '{movie}' so i am unable to help. Try a different movie!\n")



while True:
    print("\nDo you want to use my movie recommender system? (please do!!)\n")
    ans=input("Y/N => ")
    match ans.lower():
        case "y":
            movie_str=input("Please enter a movie u watched and i shall recommend u more :) => ")
            recommend(movie_str)
        case "n":
            print("okk :(  see u later...\n")
            break
        case _:
            print("\nPlease say somthing I can understand!!!\n")
    