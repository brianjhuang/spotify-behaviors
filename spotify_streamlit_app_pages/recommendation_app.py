import streamlit as st
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

global s
spotify_image_left, spotify_image_right = st.columns([1,8])

with spotify_image_left:
	spotify_logo = st.image("../spotify_streamlit_photos/spotify.png")

st.markdown("""
	# Recommendation Demo
	""")



userOneFeatures = pd.read_csv("../userOneFeatures.csv")
userTwoFeatures = pd.read_csv("../userTwoFeatures.csv")
userThreeFeatures = pd.read_csv("../userThreeFeatures.csv")

userOneFeatures.drop('Unnamed: 0', axis = 1, inplace = True)
userTwoFeatures.drop('Unnamed: 0', axis = 1, inplace = True)
userThreeFeatures.drop('Unnamed: 0', axis = 1, inplace = True)

update = st.button("Wanna change the songs? Hit this!")
#the top 50 songs is always changing, so this lets us fetch new songs

valid = False

if update:
	if os.path.exists('X_features.json'):
		os.remove('X_features.json')
	if os.path.exists('X_songs.txt'):
		os.remove('X_songs.txt')
	if os.path.exists('y_features.json'):
		os.remove('y_features.json')
	if os.path.exists('y_songs.txt'):
		os.remove('y_songs.txt')


user = st.empty()
u = user.text_input("Enter client ID: ")
if u == "":
	user = st.empty()
password = st.empty()
p = password.text_input("Enter client secret: ")
if p == "":
	password = st.empty()
if len(u) > 0 and len(p) > 0:
	s = Spotify(u, p)
	if s.perform_auth():
		st.write('Authentication Passed')
		valid = True
	else:
		st.write('Authentication Failed')
		valid = False


if valid:
	def get_X(playlist = 'Top 50 - USA', creator = 'Spotify', playlist_id = ""):
		'''
		Takes in a playlis,m ;l909t name and the creator of the playlist and
		returns the list of songs in that playlist. Also returns the features.

		Params:
		playlist (string) - the name of the playlist (default Top 50)
		creator (string) - the creator of the playlist (default Spotify)

		Returns:
		songs (list) - a list of song names
		X (list of dictionaries) - our features
		'''
		if os.path.exists('X_songs.txt'):
			#if our song exists
			songs = open('X_songs.txt').read().split("~")
			if len(songs) <= 1:
				songs = s.get_playlist_items(query = playlist, desired_artist = creator, playlist_id = playlist_id)
				with open('X_songs.txt', 'w', encoding = 'utf-8') as s_file:
					for song in songs:
						s_file.write(song[0] + "~")
					songs = open('X_songs.txt').read().split("~")
		else:
			songs = s.get_playlist_items(query = playlist, desired_artist = creator, playlist_id = playlist_id)
			with open('X_songs.txt', 'w', encoding = 'utf-8') as s_file:
				for song in songs:
					s_file.write(song[0] + "~")
				songs = open('X_songs.txt').read().split("~")

		if os.path.exists('X_features.json'):
			f = open('X_features.json')
			features = json.load(f)
			if len(features) <= 1:
				features = s.get_playlist_features(query = playlist, desired_artist = creator, playlist_id = playlist_id)
				with open('X_features.json', 'w', encoding='utf-8') as f:
					json.dump(features, f, ensure_ascii=False, indent=4)
		else:
			features = s.get_playlist_features(query = playlist, desired_artist = creator, playlist_id = playlist_id)
			with open('X_features.json', 'w', encoding='utf-8') as f:
				json.dump(features, f, ensure_ascii=False, indent=4)

		return songs[:len(songs)-1], features

	def get_y(playlist, creator, playlist_id):
		'''
		Takes in a playlist name and the creator of the playlist and
		returns the list of songs in that playlist. Also returns the features.

		Params:
		playlist (string) - the name of the playlist
		creator (string) - the creator of the playlist

		Returns:
		songs (list) - a list of song names
		features (list of dictionaries) - our y features
		'''
		if os.path.exists('y_songs.txt'):
			#if our song exists
			songs = open('y_songs.txt').read().split("~")
			if len(songs) <= 1:
				songs = s.get_playlist_items(query = playlist, desired_artist = creator, playlist_id = playlist_id)
				with open('y_songs.txt', 'w', encoding = 'utf-8') as s_file:
					for song in songs:
						s_file.write(song[0] + "~")
					songs = open('y_songs.txt').read().split("~")
		else:
			songs = s.get_playlist_items(query = playlist, desired_artist = creator, playlist_id = playlist_id)
			with open('y_songs.txt', 'w', encoding = 'utf-8') as s_file:
				for song in songs:
					s_file.write(song[0] + "~")
				songs = open('y_songs.txt').read().split("~")

		if os.path.exists('y_features.json'):
			f = open('y_features.json')
			features = json.load(f)
			if len(features) <= 1:
				features = s.get_playlist_features(query = playlist, desired_artist = creator, playlist_id = playlist_id)
				with open('y_features.json', 'w', encoding='utf-8') as f:
					json.dump(features, f, ensure_ascii=False, indent=4)
		else:
			features = s.get_playlist_features(query = playlist, desired_artist = creator, playlist_id = playlist_id)
			with open('y_features.json', 'w', encoding='utf-8') as f:
				json.dump(features, f, ensure_ascii=False, indent=4)

		return songs[:len(songs)-1], features
	playlist_valid = False
	playlist = st.text_input("What playlist do you want? Can't decide? Type Top 50 for the Top 50! ")
	creator = st.text_input("Who made the playlist? Can't decide? Type Spotify for the Top 50!")
	playlistId = st.text_input('Can\'t find the name? Use your ID! Find the ID in your Spotify Account! ')
	if (len(playlist) > 0 and len(creator) > 0) or (playlist == 'Top 50' and creator == 'Spotify'):
		st.write('Getting your songs...')
		playlist_valid = True
	if playlist_valid:
		X_songs, X = get_X(playlist, creator, playlistId)
		while len(X_songs) <= 1:
			X_songs, X = get_X(playlist, creator, playlistId)
		st.write("Songs retrieved!")

	playlist_valid = False

	playlist = st.text_input("What playlist do you want? ")
	creator = st.text_input("Who made the playlist? ")
	playlistId = st.text_input('Can\'t find the name? Use your ID! ')
	if len(playlist) > 0 and len(creator) > 0:
		playlist_valid = True
	if playlist_valid:
		y_songs, y = get_y(playlist, creator, playlistId)
		st.write('Getting your songs..')
		while len(y_songs) <= 1:
			y_songs, y = get_y(playlist, creator, playlistId)
		st.write("Your songs have been retrieved!")

		songOutput = st.selectbox('What song would you like to view?', tuple(y_songs))

		from recommender import songRecommender

		#plug in the user we want, the song we want
		model = songRecommender(X, y, X_songs, y_songs)
		modelOutput = model.similar(model.getX(), model.getY())

		output = modelOutput[songOutput]
		track_uri = output[0]
		songRecs = output[1]

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

		st.write('The 5 most similiar songs: ')
		sim_scores = []
		uris = []
		for rec in songRecs:
			sim_scores.append(rec[0])
			uris.append(rec[2])
		st.write('Similarity Score: ' + str(sim_scores[0]))
		st.components.v1.html(f'<iframe src="https://open.spotify.com/embed/track/{uris[0][14:]}" width="300" height="80" frameborder="0" allowtransparency="true" allow="autoplayt; encrypted-media"></iframe>', width=None, height=None, scrolling=False)
		st.write('Similarity Score: ' + str(sim_scores[1]))
		st.components.v1.html(f'<iframe src="https://open.spotify.com/embed/track/{uris[1][14:]}" width="300" height="80" frameborder="0" allowtransparency="true" allow="autoplayt; encrypted-media"></iframe>', width=None, height=None, scrolling=False)
		st.write('Similarity Score: ' + str(sim_scores[2]))
		st.components.v1.html(f'<iframe src="https://open.spotify.com/embed/track/{uris[2][14:]}" width="300" height="80" frameborder="0" allowtransparency="true" allow="autoplayt; encrypted-media"></iframe>', width=None, height=None, scrolling=False)
		st.write('Similarity Score: ' + str(sim_scores[3]))
		st.components.v1.html(f'<iframe src="https://open.spotify.com/embed/track/{uris[3][14:]}" width="300" height="80" frameborder="0" allowtransparency="true" allow="autoplayt; encrypted-media"></iframe>', width=None, height=None, scrolling=False)
		st.write('Similarity Score: ' + str(sim_scores[4]))
		st.components.v1.html(f'<iframe src="https://open.spotify.com/embed/track/{uris[4][14:]}" width="300" height="80" frameborder="0" allowtransparency="true" allow="autoplayt; encrypted-media"></iframe>', width=None, height=None, scrolling=False)

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

		with music_bar_left:
			st.image("../spotify_streamlit_photos/back_button_spotify.png", use_column_width = True)
