#MADE_similar_films

Demo for machine learning algorithms of search and similar items

___
## Prepare venv

* create new `python3 -m venv similar_app`
* activate venv `source similar_app/bin/activate`
* install packages ` python -m pip install -r requirements.txt` 

## Prepare Data
Movies catalog and ratings taken and saved to `data` folder from:

https://grouplens.org/datasets/movielens/latest/

##### Preprocessing
for download poster images and preprocess data use: `python3 data_preprocess.py`

At the output 2 csv files `data/movies_preprocessed.csv` and `data/rating_preprocessed.csv`

## Launch the app on localhost

`python app.py `

link to site: http://127.0.0.1:5000/

## Recorded Demo

https://youtu.be/Xq7c4DX2Iek

