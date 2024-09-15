# Anime Recommendation System

## Overview
This project is an Anime Recommendation System that suggests anime titles based on a selected anime. It uses a hybrid recommendation approach, combining content-based and collaborative filtering methods to provide accurate and personalized recommendations.

## Features
- **Anime Selection**: Users can select an anime from a dropdown list.
- **Anime Details**: Displays the selected anime's image and synopsis.
- **Recommendations**: Provides a list of recommended anime titles based on the selected anime, displayed as clickable cards that open the anime's MyAnimeList (MAL) page in a new tab.

## Libraries Used
- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations.
- **streamlit**: For creating the web application interface.
- **scikit-learn**: For implementing the Nearest Neighbors algorithm used in collaborative filtering.
- **scipy**: For creating sparse matrices used in collaborative filtering.
- **fuzzywuzzy**: For string matching and finding the closest anime title.
- **pickle**: For loading precomputed data and models.
- **html**: For unescaping HTML entities in anime titles.

## How It Works
1. **Data Loading**: Loads anime and rating data from CSV files.
2. **Data Preprocessing**: Converts data types and unescapes HTML entities in anime titles.
3. **Model Loading**: Loads precomputed data and models using `pickle`.
4. **Anime Finder**: Uses `fuzzywuzzy` to find the closest match for a given anime title.
5. **Content-Based Recommendations**: Uses cosine similarity to find similar anime titles based on content.
6. **Collaborative Filtering Recommendations**: Uses the Nearest Neighbors algorithm to find similar anime titles based on user ratings.
7. **Hybrid Recommendations**: Combines content-based and collaborative filtering recommendations to provide a final list of recommended anime titles.
8. **Display Recommendations**: Displays the recommended anime titles as clickable cards in the Streamlit web application.

## Usage
1. Run the Streamlit application:
   ```bash
   streamlit run app.py
2. Select an anime from the dropdown list.
3. View the selected anime’s details.
4. Click the “Show Recommendations” button to view recommended anime titles.

## Requirements
To install the required libraries, create a requirements.txt file with the following content:
```
fuzzywuzzy==0.18.0
numpy==2.1.1
pandas==2.2.2
scikit_learn==1.5.2
streamlit==1.38.0

```

Then, install the libraries using:
```
pip install -r requirements.txt
```

## Conclusion
This Anime Recommendation System provides personalized anime recommendations using a hybrid approach. It utilizes various libraries for data manipulation, numerical operations, machine learning, and web application development to deliver an efficient and user-friendly experience.

## Check it out here [Website](https://conwenu-hybrid-anime-recommendation-system-project-app-vngo5f.streamlit.app/)
