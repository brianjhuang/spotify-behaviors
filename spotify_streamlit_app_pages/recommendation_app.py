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
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

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

def auth(client_id, client_secret):
    s = Spotify(client_id, client_secret)
    return s

if not (os.path.exists('features.json') and os.path.exists('songs.txt')):
    user = st.text_input("Client ID: ")
    password = st.text_input("Client Secret: ")
    s = auth(user, password)
st.write('Authentication Passed')

st.write('Getting the Top 50 - USA...')

@st.cache
def get_songs():
    if os.path.exists('songs.txt'):
        songs = open('songs.txt').read().split(",")
        if len(songs) <= 0:
            songs = s.get_playlist_items()
            with open('songs.txt', 'w', encoding = 'utf-8') as s_file:
                for song in songs:
                    s_file.write(song + ",")
                songs = open('songs.txt').read().split(",")
            return songs
        return songs
    else:
        songs = s.get_playlist_items()
        with open('songs.txt', 'w', encoding = 'utf-8') as s_file:
            for song in songs:
                s_file.write(song + ",")
        songs = open('songs.txt').read().split(",")
        return songs

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
# def clickSong():
#     driver = webdriver.Chrome(ChromeDriverManager().install())
#     driver.get('localhost:8501')

#     driver.switch_to.frame(driver.find_element_by_xpath("/html/body/div/div[1]/div/div/div/div/section/div/div[1]/div[13]/iframe"))
#     driver.find_element_by_xpath("//*[@id=\"main\"]/div/div/div[1]/div[1]/div/div/button/svg/path").click()
# clickSong()
st.write("Here is the most simliar song from User " + str(userDown) + ": ", str(simAndSong[1]))
st.write("Here is our similarity score:", str(simAndSong[0]))

st.markdown('#')
st.markdown('#')

bar_leftspacer, music_bar_left, music_bar, music_bar_right, bar_rightspacer = st.columns([10,1.5,1.5,1.5,10])

with music_bar:
	play_button = st.image("../spotify_streamlit_photos/spotify_play_button.png")
	# if play_button:
	# 	play_button = st.image("pause_button.png")
with music_bar_right:
	st.image("../spotify_streamlit_photos/skip_button_spotify.png", use_column_width = True)
st.progress(85)
