import re
import urllib.request

import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup


def find_year(title):
    matches = re.findall(r'(\d+)', title)
    if matches:
        return matches[-1]


def find_poster_url(movie_url):
    try:
        with urllib.request.urlopen(movie_url) as response:
            html = response.read()
            soup = BeautifulSoup(html, 'html.parser')
            image_url = soup.find('div', class_='poster').a.img['src']
            extension = 'jpg'
            image_url = ''.join(image_url.partition('_')[0]) + extension
            return image_url
    except:
        print(f'Failed with {movie_url}')


tqdm.pandas()
url_df = pd.read_csv('data/links.csv', dtype={'movieId': int, 'imdbId': str}, usecols=range(2))
url_df['movie_url'] = 'http://www.imdb.com/title/tt' + url_df['imdbId'] + '/'
url_df['img_url'] = url_df['movie_url'].progress_apply(lambda x: find_poster_url(x))

movies_df = pd.read_csv('data/movies.csv')
movies_df['year'] = movies_df['title'].apply(lambda x: find_year(x))
movies_df = movies_df.merge(url_df, on='movieId')
movies_df = movies_df.dropna()
movies_df['year'] = movies_df['year'].astype(int)

ratings_df = pd.read_csv('data/ratings.csv')
ratings_df = ratings_df.merge(movies_df[['movieId']], on='movieId')

movies_df.drop('imdbId', axis=1).to_csv('data/movies_preprocessed.csv', index=False)
ratings_df.drop('timestamp', axis=1).to_csv('data/rating_preprocessed.csv', index=False)
