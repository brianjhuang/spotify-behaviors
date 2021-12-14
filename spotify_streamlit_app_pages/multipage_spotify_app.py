import streamlit as st
from multipage_template import save, MultiPage, start_app, clear_cache
import pandas as pd
from PIL import Image
import os

import numpy as np
import streamlit.components.v1 as components

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import random
from sklearn.decomposition import PCA
import math
from spotifyAPI import Spotify
import json
st.set_page_config(layout='wide')

start_app() #Clears the cache when the app is started

app = MultiPage()
app.start_button = "Let's go!"
app.navbar_name = "Navigation"
app.next_page_button = "Next Page"
app.previous_page_button = "Previous Page"


def startpage():
	spotify_image_left, spotify_image_right = st.columns([1,8])

	with spotify_image_left:
		spotify_logo = st.image("../spotify_streamlit_photos/spotify.png")

	intro_left, intro_right = st.columns([3,1])

	with intro_left:
		st.markdown("#  Will You Skip The Next Song?")
		st.write("This project focues on analyzing the recommender systems among \
			Spotify's application, a popular music streaming service. To determine\
			 how Spotify is able to recommend songs to their listeners, we will \
			 look at specific musical traits among tracks that listeners \
			 have listened to, in order to predict whether they will play \
			 through or skip a particular song.")


	st.markdown('#')
	st.markdown('#')
	st.markdown('#')
	st.markdown('#')
	st.markdown('#')


	bar_leftspacer, music_bar_left, music_bar, music_bar_right, bar_rightspacer = st.columns([10,1.5,1.5,1.5,10])

	with music_bar:
		play_button = st.image("../spotify_streamlit_photos/spotify_play_button.png")
		# if play_button:
		# 	play_button = st.image("pause_button.png")
	with music_bar_right:
		st.image("../spotify_streamlit_photos/skip_button_spotify.png", use_column_width = True)
	st.progress(2)


def landing(prev_vars): #home page
		spotify_image_left, spotify_image_right = st.columns([1,8])

		with spotify_image_left:
			spotify_logo = st.image("../spotify_streamlit_photos/spotify.png")

		intro_left, intro_right = st.columns([3,1])

		with intro_left:
			st.markdown("#  Will You Skip The Next Song?")
			st.write("This project focues on analyzing the recommender systems among \
				Spotify's application, a popular music streaming service. To determine\
				 how Spotify is able to recommend songs to their listeners, we will \
				 look at specific musical traits among tracks that listeners \
				 have listened to, in order to predict whether they will play \
				 through or skip a particular song.")


		st.markdown('#')
		st.markdown('#')
		st.markdown('#')
		st.markdown('#')
		st.markdown('#')


		bar_leftspacer, music_bar_left, music_bar, music_bar_right, bar_rightspacer = st.columns([10,1.5,1.5,1.5,10])

		with music_bar:
			play_button = st.image("../spotify_streamlit_photos/spotify_play_button.png")
			# if play_button:
			# 	play_button = st.image("pause_button.png")
		with music_bar_right:
			st.image("../spotify_streamlit_photos/skip_button_spotify.png", use_column_width = True)
		st.progress(2)



def eda(prev_vars): #EDA / Data Cleaning
	spotify_image_left, spotify_image_right = st.columns([1,8])

	with spotify_image_left:
		spotify_logo = st.image("../spotify_streamlit_photos/spotify.png")

	#EDA / Data Cleaning
	st.markdown("# EDA / Data Cleaning")

	#EDA / Data Cleaning
	st.markdown("## __Data Cleaning__")


	st.markdown("""
	### Let's first look at our data...

	With a total of 500 gb of user data, we sampled our data using Pyspark and \
	split samples into training and test sets to apply our models.

		""")

	spark_left, spark_right = st.columns(2)

	with spark_left:
		st.image("../spotify_streamlit_photos/pyspark_screenshot.jpg")

	with spark_right:
		st.write("This is code for how we came up with the samples in pyspark.")
		st.write("1) import all required packages")
		st.write("2) create a new spark session")
		st.write("3) load the dataframe so we can sample it")
		st.write("4) convert to work with pandas")

	#importing data and combining into one dataframe
	st.markdown("### Shown below is the sampled dataset...")

	log_data = pd.read_csv("../data/training_set/log_mini.csv")
	track_data = pd.read_csv("../data/track_features/tf_mini.csv")
	log_data = log_data.rename(columns = {'track_id_clean':'track_id'})

	df = pd.merge(log_data,track_data,on='track_id',how='left')

	#display the dataframe
	st.write(df.head(100))

	#display the columns of track and log data
	log_left, track_right = st.columns(2)

	with log_left:
		st.markdown("### Log Data Columns")
		st.write(log_data.columns)

	with track_right:
		st.markdown("### Track Data Columns")
		st.write(track_data.columns)

	link = st.expander("To better understand the meaning of each column, follow this link.")
	link.write('https://drive.google.com/file/d/1aR6g0hGhue3pGZ81buEXvRCFLmdMfFep/view?usp=sharing')


	st.write("Among the Log Data, skip results are given by the columns 'not_skipped', \
		'skip_1','skip_2', and 'skip_3'. These columns are combined into one column with boolean values that \
		shows whether the user has skipped the song.")

	st.write("The column 'hist_user_behavior_reason_end' directly determines users' skip behavior. As a result, \
	it is dropped for our model to better predict the outcomes without such giveaways. ")

	#get_skip
	st.markdown("""## __EDA__
	We performed exploratory data analysis to explore the patterns between different features \
	and users' skipping behavior""")

	st.markdown("#### Overall Skip Behavior")
	skip_left, skip_right = st.columns(2)

	with skip_left:
		st.image("../spotify_streamlit_photos/skip_eda.png")
	with skip_right:
		st.write("In our sample dataset, there are 111996 skipped entries and 55884 not_skipped entries.")

	st.markdown("#### pause_before_play vs. Skip Behavior")
	pause_left, pause_right = st.columns(2)
	with pause_left:
		st.image("../spotify_streamlit_photos/pause_eda.png")
		st.caption("True percentage of users not skipping the current song, grouped by how long of a pause \
		the user takes before playing the current track.")
	with pause_right:
		st.image("../spotify_streamlit_photos/pause_eda_plot.png")
		st.caption("Boxplot showing the number of users who skipped and not skipped grouped by how long of a pause \
		the user takes before playing the current track.")

	st.markdown("#### Premium vs. Skip Behavior")
	premium_left, premium_right = st.columns(2)
	with premium_left:
		st.image("../spotify_streamlit_photos/premium_eda.png")
		st.caption("True percentage of users not skipping the current song, grouped by whether the user is premium or not.")
	with premium_right:
		st.image("../spotify_streamlit_photos/premium_eda_plot.png")
		st.caption("Boxplot showing the number of users who skipped and not skipped grouped by whether the user is premium or not.")

	st.markdown("#### Shuffle vs. Skip Behavior")
	shuffle_left, shuffle_right = st.columns(2)
	with shuffle_left:
		st.image("../spotify_streamlit_photos/shuffle_eda.png")
		st.caption("True percentage of users not skipping the current song, grouped by whether the user is in shuffle mode.")
	with shuffle_right:
		st.image("../spotify_streamlit_photos/shuffle_eda_plot.png")
		st.caption("Boxplot showing the number of users who skipped and not skipped grouped by whether the user is in \
		shuffle mode or not.")

	st.markdown("""
	### Tracks' features vs. Skip Behavior
		""")
	# def get_skip(df):
	#     if df['not_skipped'] == 1:
	#         return 'not skipped'
	#     else:
	#         return 'skipped'
	# skip_info = df.apply(get_skip, axis = 1)
	# df = df.assign(skip_type = skip_info)

	duration_left, duration_right = st.columns(2)

	with duration_left:
		st.image("../spotify_streamlit_photos/danceability_boxplot.jpg")
		st.caption("Boxplot showing the relationship between a track's danceability and users' skip behavior.")

	with duration_right:
		st.image("../spotify_streamlit_photos/duration_boxplot.jpg")
		st.caption("Boxplot showing the relationship between a track's duration and users' skip behavior")

	bar_leftspacer, music_bar_left, music_bar, music_bar_right, bar_rightspacer = st.columns([10,1.5,1.5,1.5,10])

	with music_bar:
		play_button = st.image("../spotify_streamlit_photos/spotify_play_button.png")
		# if play_button:
		# 	play_button = st.image("pause_button.png")
	with music_bar_right:
		st.image("../spotify_streamlit_photos/skip_button_spotify.png", use_column_width = True)

	with music_bar_left:
		st.image("../spotify_streamlit_photos/back_button_spotify.png", use_column_width = True)
	st.progress(20)



def model(prev_vars):
	spotify_image_left, spotify_image_right = st.columns([1,8])

	with spotify_image_left:
	  spotify_logo = st.image("../spotify_streamlit_photos/spotify.png")


	st.markdown('# Spotify Behavior Model')

	#BASELINE sklearn model, FEEL FREE TO EDIT
	log_data = pd.read_csv("../data/training_set/log_mini.csv")
	track_data = pd.read_csv("../data/track_features/tf_mini.csv")
	log_data = log_data.rename(columns = {'track_id_clean':'track_id'})

	df = pd.merge(log_data,track_data,on='track_id',how='left')

	model_left, model_right = st.columns(2)

	with model_left:
	  st.markdown("## __Sklearn Model__ ")
	  st.write("Using Sklearn, a machine learning package used alongside python, we implemented \
	  Logistic Regression and Random Forest Classifier techniques to predict skip behavior \
	  given specific musical tracks.")

	with model_right:
	  st.markdown('### Interested in the code?')
	  with st.expander("Click here to expand."):
	    st.image('../spotify_streamlit_photos/model_code.jpg')


	log_data['not_skipped'] = log_data['not_skipped'].apply(lambda x: 1 if x == True else 0)

	log_data['premium']= log_data['premium'].apply(lambda x: 1 if x is True else 0)
	log_data['hist_user_behavior_is_shuffle'] = log_data['hist_user_behavior_is_shuffle'].apply(lambda x: 1 if x is True else 0)

	as_is = ['session_position', 'session_length','hist_user_behavior_is_shuffle',
	       'hour_of_day','premium','duration',
	       'release_year', 'us_popularity_estimate', 'acousticness',
	       'beat_strength', 'bounciness', 'danceability', 'dyn_range_mean',
	       'energy', 'flatness', 'instrumentalness', 'liveness', 'loudness',
	       'mechanism', 'key', 'organism', 'speechiness', 'tempo',
	       'time_signature', 'valence', 'acoustic_vector_0', 'acoustic_vector_1',
	       'acoustic_vector_2', 'acoustic_vector_3', 'acoustic_vector_4',
	       'acoustic_vector_5', 'acoustic_vector_6', 'acoustic_vector_7',
	       'context_switch', 'no_pause_before_play', 'short_pause_before_play',
	       'long_pause_before_play', 'hist_user_behavior_n_seekfwd',
	       'hist_user_behavior_n_seekback']
	ohe = ['mode','context_type', 'hist_user_behavior_reason_start']

	preproc = ColumnTransformer(
	    transformers = [
	        ('as_is', FunctionTransformer(lambda x: x), as_is),
	        ('one_hot', OneHotEncoder(handle_unknown = 'ignore'), ohe)
	    ]
	)

	to_predict = df['not_skipped']

	pl = Pipeline(steps = [('preprocessor', preproc), ('classifier', DecisionTreeClassifier(max_depth = 10))])
	x_train, x_test, y_train, y_test = train_test_split(df.drop('not_skipped', axis = 1), df['not_skipped'], test_size= 0.2)
	model = pl.fit(x_train, y_train)
	predictions = model.predict(x_test)

	predict_col1, predict_col2, predict_col3, predict_col4 = st.columns(4)

	with predict_col1:
	  predict_button = st.button("Predict")
	with predict_col2:
	  st.write("< interact with this button!")

	if predict_button:

	  sk_col1, sk_col2, sk_col3, sk_col4 = st.columns(4)

	  with sk_col1:
	    st.write(predictions)
	  with sk_col2:
	    score = pl.score(x_test, y_test)
	    st.write(score)
	    st.write("The prediction accuracy score is " + str(score) + "!")

	#spotify play area
	bar_leftspacer, music_bar_left, music_bar, music_bar_right, bar_rightspacer = st.columns([10,1.5,1.5,1.5,10])

	with music_bar:
	  play_button = st.image("../spotify_streamlit_photos/spotify_play_button.png")
	  # if play_button:
	  #   play_button = st.image("pause_button.png")
	with music_bar_right:
	  st.image("../spotify_streamlit_photos/skip_button_spotify.png", use_column_width = True)
	st.progress(40)

	with music_bar_left:
		st.image("../spotify_streamlit_photos/back_button_spotify.png", use_column_width = True)


def recommendation(pre_vars):
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



def discussion(prev_vars): #problems/problems
	spotify_image_left, spotify_image_right = st.columns([1,8])

	with spotify_image_left:
		spotify_logo = st.image("../spotify_streamlit_photos/spotify.png")

	#Title of Page
	st.markdown("# Discussion of Methods")


	#problems with sklearn
	st.markdown("""## __Experiences with scikit-learn__	""")

	st.image("../spotify_streamlit_photos/sklearn.png", width = 1000)

	st.markdown("""
	Using sklearn to generate a model that would predict skipping behavior allowed for
	data from different users to become a general prediction. In the real world, this
	is not accurate as every listener has their own unique taste and skipping behavior.
	In other words, we simply cannot assume that everyone shares the same behavior through
	sampling from the whole dataset.

	Rather, it must be recognized that each listener has their unique listening styles.
	Therefore, it is essential that we sample the listening behavior of each specific listener
	in order to determine traits that shape the skipping behavior of listeners.
		""")

	#problems with behavior class
	st.markdown("""## __The Recommendation Class__""")

	st.image("../spotify_streamlit_photos/clustering.png")

	st.markdown("""
	For our demo, recommendations were generated using Cosine similarity, taking in user data
	and grouping the songs they listened to with the user. We generated a similarity function
	to train a system that takes in a group of songs and their features, and outputs what songs
	users are most likely to listen/are most similar to what our persona users will listen to.

	However we still ran into the problem of grouping users by their listening activity.
	One of the biggest problems when making our predictions, people listen to songs in order,
	and those songs impact what they listen to next. We did not have time to implement the
	weight of song sequencing, which could've improved our predictions a ton, as knowing how
	people listen to songs provides crucial information. If we had more time, we could explore using
	Markov chains in the future to account for sequencing.

	We solved this problem of the "superuser" using k-means clustering for each user in our recommendation class.

		""")
	st.markdown("""## __Data, Privacy, and Other Issues__""")

	st.image("../spotify_streamlit_photos/privacy.jpeg")

	st.markdown("""
	Our dataset was large, far larger than any dataset any of us had worked with up this point.
	With well over 500GBs of data, our team was tasked with hours of sampling, cleaning the data
	using PySpark, and leaving our devices open overnight to extract data samples of usable size.
	Balancing the size of the sampled data while preserving the integrity of the data was a major time sink.
	Thankfully, due to PySpark and the Mac application Amphetamine (which kept our laptops awake long enough
	to sample overnight), we were able to sample workable sets of data.

	In an attempt to add a bit of spice to our project, we attempted to fetch the actual song names
	from the Spotify API. This, however, resulted in a grim discovery. The data returned, since
	it's user data, had the track IDs encoded. This meant there was no way for us to retrieve tracks.
	On top of that, that meant that many songs were encoded with unique IDs, meaning that the same song
	would have a different ID. This made building out our recommender system difficult as it introduced
	unnecessary redundancy that we couldn't resolve when clustering the data. In the future, we can avoid
	this by extracting the data ourselves, rather than relying on an existing set.

	As with all recommendations, we cannot account for users who have no prior history, or tracks
	we have never seen before. We were unable to deal with the Cold Start problem, and simply allowed
	our algorithm to give it's best guess when it came to dealing with this issue.

		""")

	st.markdown('#')
	st.markdown('#')


	bar_leftspacer, music_bar_left, music_bar, music_bar_right, bar_rightspacer = st.columns([10,1.5,1.5,1.5,10])

	with music_bar:
		play_button = st.image("../spotify_streamlit_photos/spotify_play_button.png")
		# if play_button:
		# 	play_button = st.image("pause_button.png")
	with music_bar_right:
		st.image("../spotify_streamlit_photos/skip_button_spotify.png", use_column_width = True)
	st.progress(65)

	with music_bar_left:
		st.image("../spotify_streamlit_photos/back_button_spotify.png", use_column_width = True)




def about_us(prev_vars): #About Us Page
	spotify_image_left, spotify_image_right = st.columns([1,8])

	with spotify_image_left:
		spotify_logo = st.image("../spotify_streamlit_photos/spotify.png")

	#Title of Page
	st.markdown("""## __About Our Team__""")
	st.markdown("### Hello! We are undergraduate data science students at the University of California - San Diego")



	images = ["../spotify_streamlit_photos/brian.jpg", "../spotify_streamlit_photos/annie.jpg", "../spotify_streamlit_photos/victor.jpg",
	"../spotify_streamlit_photos/aishani.jpg"]
	#st.image(images, width=300, caption=["Brian H", "Annie Fan", "Victor Thai", "Aishani Mohapatra" ])
	brian, annie, victor, aishani = st.columns(4)
	with brian:
		brian = st.image(images[0])
		link = '[Brian Huang](https://www.linkedin.com/in/brian-huang-b5238817b/)'
		st.markdown(link, unsafe_allow_html=True)
		st.write("My name is Brian and I was born June 28 in Singapore. I am currently a data science major and psychology minor at Warren College. \
		I aspire to apply my data analysis skills to biology or medicine to speed up research processes. I have two sisters, Katherine and Angelina, and I enjoy playing tennis and cycling.")
	with annie:
		annie = st.image(images[1])
		link = '[Annie Fan](https://www.linkedin.com/in/annie-fan-b45273208/)'
		st.markdown(link, unsafe_allow_html=True)
		st.write("My name is Annie Fan and I am a third year student majoring in data science and cognitive science. \
			I grew up in China and I attended high school in Seattle, Washington. I am into data analysis and machine learning. \
			Currently, I am learning data management with SQL and building recommender systems with data mining.")
	with victor:
		victor = st.image(images[2])
		link = '[Victor Thai](https://www.linkedin.com/in/victor-thai-37334317b/)'
		st.markdown(link, unsafe_allow_html=True)
		st.write("My name is Victor and I am a second year studying data science at UCSD. I grew up in Oakland, CA and have just recently moved to San Diego to pursue my studies. \
			In my free time, I enjoy staying active and being in nature by going on hikes. I plan on using data science to create improvements to heatlthcare technologies and procedures.")
	with aishani:
		aishani = st.image(images[3])
		link = '[Aishani Mohapatra](http://www.linkedin.com/in/aishani-mohapatra)'
		st.markdown(link, unsafe_allow_html=True)
		st.write("My name is Aishani and I am a second year at Muir College studying data science from San Ramon, CA. I am interested in exploring topics the intersection between language and machine learning, \
			 particularly with Natural Language Processing. In my free time, I enjoy singing and hiking with friends.")

	st.markdown('#')
	st.markdown('#')

	bar_leftspacer, music_bar_left, music_bar, music_bar_right, bar_rightspacer = st.columns([10,1.5,1.5,1.5,10])

	with music_bar:
		play_button = st.image("../spotify_streamlit_photos/spotify_play_button.png")
		# if play_button:
		# 	play_button = st.image("pause_button.png")
	with music_bar_right:
		st.image("../spotify_streamlit_photos/skip_button_spotify.png", use_column_width = True)
	st.progress(90)

	with music_bar_left:
		st.image("../spotify_streamlit_photos/back_button_spotify.png", use_column_width = True)



app.set_initial_page(startpage) #home/landing page
app.add_app("Home", landing) #home/landing page
app.add_app("EDA", eda) #Adds second page (eda) to the framework
app.add_app("Model", model) #Adds third page (model) to the framework
app.add_app("Recommendation", recommendation)
app.add_app("Discussion", discussion) #Adds fourth page (discussion) to the framework
app.add_app("About Us", about_us) #Adds last page (about us) to the framework

app.run() #Runs the multipage app!
