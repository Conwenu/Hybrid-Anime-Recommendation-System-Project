import pandas as pd
import numpy as np
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
import pickle
import html

# Load data
filtered_animes = pd.read_csv('filtered_animes.csv')
filtered_animes['animeId'] = pd.to_numeric(filtered_animes['animeId'], errors='coerce')
filtered_animes['image_link'] = filtered_animes['image_link'].astype(str) 
filtered_animes['title'] = filtered_animes['title'].apply(html.unescape)
filtered_animes['mal_url'] = filtered_animes['mal_url'].astype(str)  # Ensure mal_url is string type

# Load pickled objects
with open('anime_inv_mapper.pkl', 'rb') as file:
    anime_inv_mapper = pickle.load(file)

with open('user_inv_mapper.pkl', 'rb') as file:
    user_inv_mapper = pickle.load(file)

with open('user_mapper.pkl', 'rb') as file:
    user_mapper = pickle.load(file)

with open('anime_mapper.pkl', 'rb') as file:
    anime_mapper = pickle.load(file)

with open('cosine_similarity_matrix.pkl', 'rb') as file:
    cosine_sim = pickle.load(file)

# Helper functions
def anime_finder(title):
    all_titles = filtered_animes['title'].tolist()
    closest_match = process.extractOne(title, all_titles)
    return closest_match[0], closest_match[1]

def get_item_based_recommendations(anime_id):
    with open('CSRMatrix.pkl', 'rb') as file:
        X = pickle.load(file)

    X = X.T
    neighbour_ids = []
    scores = []
    
    anime_ind = anime_mapper[anime_id]
    anime_vec = X[anime_ind]
    if isinstance(anime_vec, (np.ndarray)):
        anime_vec = anime_vec.reshape(1, -1)
    
    kNN = NearestNeighbors(n_neighbors=X.shape[0], algorithm="brute", metric='cosine')
    kNN.fit(X)
    
    neighbour = kNN.kneighbors(anime_vec, return_distance=True)
    for i in range(1, X.shape[0]):  # Skip the given anime itself
        n = neighbour[1].item(i)
        neighbour_ids.append(anime_inv_mapper[n])
        score = 1 - neighbour[0].item(i)
        scores.append(score)
    
    similar_animes_df = pd.DataFrame({
        'animeId': neighbour_ids,
        'score': scores
    })
    
    similar_animes_df = similar_animes_df.merge(filtered_animes[['animeId', 'title']], on='animeId', how='left')
    
    return similar_animes_df

def getIndexFromTitle(title):
    anime_idx = dict(zip(filtered_animes['title'], list(filtered_animes.index)))
    idx = anime_idx[title]
    return idx

def getAnimeIdFromTitle(title):    
    title, percent = anime_finder(title)
    if title not in filtered_animes['title'].values:
        return None
    title_to_id = dict(zip(filtered_animes['title'], filtered_animes['animeId']))
    anime_id = title_to_id.get(title)
    return anime_id

def get_content_based_recommendations(title_string):
    title, percent = anime_finder(title_string)
    if percent < 75:
        print(f"The Similarity Score for '{title_string}' to '{title}' is {percent}%.\nPlease note that the title extracted may not be what you're looking for.")
    
    if title is None:
        return
    
    idx = getIndexFromTitle(title)
    if idx is None:
        return
    
    if idx >= len(cosine_sim):
        return
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:]
    
    similar_animes = [i[0] for i in sim_scores]
    distances = [i[1] for i in sim_scores]
    
    if any(i >= len(filtered_animes) for i in similar_animes):
        return
    
    recommendations_df = pd.DataFrame({
        'animeId': filtered_animes['animeId'].iloc[similar_animes].values,
        'title': filtered_animes['title'].iloc[similar_animes].values,
        'distance': [1 - score for score in distances]  # Convert similarity to distance
    })
        
    return recommendations_df

def get_hybrid_recommendations(title, top_n = 10):  
    extractedTitle, percent = anime_finder(title)
    extractedId = getAnimeIdFromTitle(extractedTitle)
    
    content_based_recommendations = get_content_based_recommendations(extractedTitle)
    collaborative_filtering_recommendations = get_item_based_recommendations(extractedId)
    
    combined_recommendations = pd.merge(content_based_recommendations, collaborative_filtering_recommendations, on='animeId', suffixes=('', '#'))
    combined_recommendations.drop(columns=['title#'], inplace=True)
    
    combined_recommendations['normalized_score'] = combined_recommendations['score']
    combined_recommendations['normalized_distance'] = 1 - combined_recommendations['distance']
    
    weight_score = 0.7
    weight_distance = 0.3 

    combined_recommendations = pd.merge(
        combined_recommendations,
        filtered_animes[['animeId', 'image_link', 'mal_url']],
        on='animeId',
        how='left'
    )
    
    combined_recommendations['new_score'] = (weight_score * combined_recommendations['normalized_score']) + (weight_distance * combined_recommendations['normalized_distance'])

    combined_recommendations = combined_recommendations.sort_values(by='new_score', ascending=False)
    combined_recommendations.drop(columns=['normalized_score', 'normalized_distance', 'score', 'distance'], inplace=True)
    
    top_recommendations = combined_recommendations.sort_values(by='new_score', ascending=False).head(top_n)
    return top_recommendations

def display_anime_details(selected_anime):
    selected_anime_data = filtered_animes[filtered_animes['title'] == selected_anime]
    if not selected_anime_data.empty:
        image_url = selected_anime_data['image_link'].values[0]
        synopsis = selected_anime_data['summary'].values[0]
        
        st.image(image_url, caption=selected_anime, width=150)
        st.subheader('Synopsis:')
        st.write(synopsis)

def display_recommendations_as_cards(selected_anime):
    if st.button("Show Recommendations"):
        st.subheader(f"Because you liked '{selected_anime}', here are some recommendations:")
        recommendations = get_hybrid_recommendations(selected_anime, top_n=15)
        
        num_columns = 5
        num_rows = (len(recommendations) + num_columns - 1) // num_columns
        
        for row_index in range(num_rows):
            with st.container():
                cols = st.columns(num_columns)
                for col_index in range(num_columns):
                    item_index = row_index * num_columns + col_index
                    if item_index < len(recommendations):
                        row = recommendations.iloc[item_index]
                        with cols[col_index]:
                            card_html = f"""
                            <a href="{row['mal_url']}" target="_blank" style="text-decoration: none; outline: none;">
                                <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; text-align: center;">
                                    <img src="{row['image_link']}" style="width: 100%; border-radius: 5px;">
                                    <p style="font-weight: bold; color: white;">{row['title']}</p>
                                </div>
                            </a>
                            """
                            st.markdown(card_html, unsafe_allow_html=True)
                    else:
                        with cols[col_index]:
                            st.write("")

def main():
    st.title("Anime Recommendation System")
    
    selected_anime = st.selectbox('Select an anime:', filtered_animes['title'])
    display_anime_details(selected_anime)
    display_recommendations_as_cards(selected_anime)

if __name__ == "__main__":
    main()
