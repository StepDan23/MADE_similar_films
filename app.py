import numpy as np
import pandas as pd
from scipy import sparse

from flask import render_template, Flask, request, redirect, url_for
from forms import SearchForm

from rank_bm25 import BM25Okapi
from implicit.als import AlternatingLeastSquares as Als

FILMS_ON_MAIN = 5
N_SIMILAR_FILMS = 10
N_SEARCH_RESULTS = 10
PREPROCESSED_MOVIES_PATH = 'data/movies_preprocessed.csv'
PREPROCESSED_RATING_PATH = 'data/rating_preprocessed.csv'


def make_sparse():
    data_df = pd.read_csv(PREPROCESSED_RATING_PATH)
    users_col = 'userId'
    items_col = 'movieId'
    score_col = 'rating'

    data_df[users_col] = data_df[users_col].astype('category')
    data_df[items_col] = data_df[items_col].astype('category')
    sparse_coo = sparse.coo_matrix((data_df[score_col], (data_df[users_col].cat.codes, data_df[items_col].cat.codes)))
    items_convert_dict = dict(enumerate(data_df[items_col].cat.categories))
    return sparse_coo.tocsr(), items_convert_dict


def preprocess_text(title_name):
    words = title_name.lower().replace('(', '').replace(')', '').split()
    words_with_prefix = [word[:i] for word in words for i in range(3, len(word) + 1)]
    return words_with_prefix


similar_model = Als(factors=50)
ratings_csr, model_real_indices = make_sparse()
real_model_indices = {real_ind: model_ind for model_ind, real_ind in model_real_indices.items()}
similar_model.fit(ratings_csr.T)

all_movies_df = pd.read_csv(PREPROCESSED_MOVIES_PATH, index_col=0)
search_corpus = [preprocess_text(title) for title in all_movies_df['title']]
search_model = BM25Okapi(search_corpus)

app = Flask('movies')
app.config['SECRET_KEY'] = 'any secret string'


def get_similar_films(real_id):
    if real_id in real_model_indices:
        model_id = real_model_indices[real_id]
        indices = similar_model.similar_items(model_id, N=N_SIMILAR_FILMS + 1)
        real_similar_ids = [model_real_indices[ind] for ind, score in indices if ind != model_id]
        return [row for ind, row in all_movies_df.loc[real_similar_ids].iterrows()]
    else:
        return []


def get_search_result(query):
    tokens = preprocess_text(query)
    scores = search_model.get_scores(tokens)
    indices = np.nonzero(scores)[0]
    candidates = indices[np.argsort(scores[indices])[::-1][:N_SEARCH_RESULTS]]
    return [row for ind, row in all_movies_df.iloc[candidates].iterrows()]


@app.route('/')
@app.route('/index')
def index():
    random_movies = [row for ind, row in all_movies_df.sample(n=FILMS_ON_MAIN).iterrows()]
    return render_template('index.html', title='Home', random_movies=random_movies)


@app.route('/film', methods=['GET'])
def film():
    if request.args.get('id'):
        film_id = int(request.args.get('id'))
        film_attr = all_movies_df.loc[film_id]
        similar_films = get_similar_films(film_id)
        return render_template('film.html', title=film_attr['title'], film_attr=film_attr, similar_films=similar_films)
    else:
        return redirect(url_for('search'))


@app.route('/search', methods=['GET', 'POST'])
def search():
    form = SearchForm()
    if request.method == 'POST':
        if form.validate_on_submit():
            search_results = get_search_result(form.search_box.data)
            return render_template('search.html', title='Search', form=form,
                                   search_results=search_results, query=form.search_box.data)
        else:
            return redirect(url_for('search'))
    else:
        return render_template('search.html', title='Search', form=form, query=None)


if __name__ == '__main__':
    app.run()
