# recommender/views.py

from django.shortcuts import render
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from django.conf import settings

# Load the dataset and ml array
with open(settings.DATA_DIR / 'meta_games_dict.pkl', 'rb') as f:
    movies = pickle.load(f)

with open(settings.DATA_DIR / 'confusion_matrix_con.pkl', 'rb') as f:
    ml = pickle.load(f)

cosine_sim = cosine_similarity(ml, ml)
movies = pd.DataFrame(movies)
# change developer name of Counter-Strike: Global Offensive
movies.loc[movies['name'] == 'Counter-Strike: Global Offensive', 'developer'] = 'Valve'
def get_recommendations_by_tags(game_name, cosine_sim=cosine_sim, df=movies):
    idx = df[df['name'].str.lower() == game_name.lower()].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    game_indices = [i[0] for i in sim_scores]
    return df['name'].iloc[game_indices]

def get_recommendations_by_developer(game_name, df=movies):
    developer = df[df['name'].str.lower() == game_name.lower()]['developer'].values[0]
    return df[df['developer'] == developer]['name']

def get_recent_games(df=movies, top_n=10):
    recent_games = df.sort_values(by='release_date', ascending=False)
    return recent_games['name'].head(top_n)

def get_top_reviewed_games(df=movies, top_n=10):
    top_reviewed = df.sort_values(by='all_reviews', ascending=False)
    return top_reviewed['name'].head(top_n)

# recommender/views.py

def home(request):
    top_reviewed_games = get_top_reviewed_games()
    recent_games = get_recent_games()
    context = {
        'top_reviewed_games': top_reviewed_games,
        'recent_games': recent_games
    }
    return render(request, 'home.html', context)

def recommend(request):
    if request.method == 'POST':
        game = request.POST['game']
        tag_recommendations = get_recommendations_by_tags(game)
        developer_recommendations = get_recommendations_by_developer(game)
        recent_games = get_recent_games()
        top_reviewed_games = get_top_reviewed_games()

        context = {
            'game': game,
            'tag_recommendations': tag_recommendations,
            'developer_recommendations': developer_recommendations,
            'recent_games': recent_games,
            'top_reviewed_games': top_reviewed_games,
        }
        return render(request, 'recommender.html', context)
    return render(request, 'home.html')
