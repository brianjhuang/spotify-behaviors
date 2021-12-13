import streamlit as st
from multipage_template import save, MultiPage, start_app, clear_cache
import pandas as pd
from PIL import Image
import os
# from pyspark.sql import functions as f
# from pyspark.sql import SparkSession
import random
from sklearn.decomposition import PCA
import math
from spotifyAPI import Spotify
import json

spotify_image_left, spotify_image_right = st.columns([1,8])

with spotify_image_left:
	spotify_logo = st.image("../spotify_streamlit_photos/spotify.png")

st.markdown("""
	# Recommendation Demo
	""")

st.write("For our recommendation model, we have generated it using cosine similarity. \
	We first took in user data and grouped songs that they listened to. Using a similarity function, \
	we used this function to train a system that takes in a group of songs and its features to \
	return songs that these users are similar to the songs that they have already listened to. ")

st.markdown("""
	## Similarity Function

	To determine the relative similarities that each song has with our prediction set, we used the following similarity function.
	""")

userOneFeatures = pd.read_csv("../userOneFeatures.csv")
userTwoFeatures = pd.read_csv("../userTwoFeatures.csv")
userThreeFeatures = pd.read_csv("../userThreeFeatures.csv")

userOneFeatures.drop('Unnamed: 0', axis = 1, inplace = True)
userTwoFeatures.drop('Unnamed: 0', axis = 1, inplace = True)
userThreeFeatures.drop('Unnamed: 0', axis = 1, inplace = True)

s = Spotify()
st.write('Authentication Passed')

st.write('Getting the Top 50 - USA...')

@st.cache
def get_songs():
    return s.get_playlist_items()

@st.cache
def run_model(data, songs):
    if os.path.exists('features.json'):
        f = open('features.json')
        features = json.load(f)
        if len(features) <= 0:
            features = s.get_playlist_features('Top 50 - USA')
            with open('features.json', 'w', encoding='utf-8') as f:
                json.dump(features, f, ensure_ascii=False, indent=4)
        model = songRecommender(data = userDict[userDown], predict = features, songs = songs)
    else:
        features = s.get_playlist_features('Top 50 - USA')
        with open('features.json', 'w', encoding='utf-8') as f:
            json.dump(features, f, ensure_ascii=False, indent=4)
        model = songRecommender(data = userDict[userDown], predict = features, songs = songs)
    return model

songs = get_songs()
st.write("Top 50 Songs Retrieved!")
songOutput = st.selectbox('What song would you like to view?', tuple(songs))

from songRecommender import songRecommender

userTuple = ('One', 'Two', "Three")

userDown = st.selectbox('What user would you like to pick?', userTuple)
userDict = {'One': userOneFeatures, 'Two': userTwoFeatures, 'Three': userThreeFeatures}

#plug in the user we want, the song we want
model = run_model(userDict[userDown], songs)
modelOutput = model.similar(model.getPredict(), model.getFeatures())

output = modelOutput[songOutput]
track_uri = output[0]
simAndSong = output[1][0]

uri = f'<iframe src="https://open.spotify.com/embed/track/{track_uri[14:]}" width="300" height="80" frameborder="0" allowtransparency="true" allow="autoplayt; encrypted-media"></iframe>'
html_string ='''
<script language = 'javascript'>
    function autoplay() {
        var t = setTimeout(function(){
            var button = document.querySelector('[title="Play"]') || false;
            if (button) {
                console.log('Click', button)
                button.click()
            }
        }, 999)
    }
    document.addEventListener('DOMContentLoaded', (event) => {
        autoplay()
    })
</script>
'''

st.write("The song you chose:")
st.components.v1.html(uri + html_string, width=None, height=None, scrolling=False)


st.write("Here is the most simliar song from User " + str(userDown) + ": ", str(simAndSong[1]))
st.write("Here is our similarity score:", str(simAndSong[0]))



