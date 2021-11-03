import streamlit as st
from multipage import save, MultiPage, start_app, clear_cache
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
		spotify_logo = st.image("spotify.png")

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
		play_button = st.image("spotify_streamlit_photos/spotify_play_button.png")
		# if play_button:
		# 	play_button = st.image("pause_button.png")
	with music_bar_right:
		st.image("spotify_streamlit_photos/skip_button_spotify.png", use_column_width = True)
	st.progress(2)


def eda(prev_vars): #EDA / Data Cleaning
	spotify_image_left, spotify_image_right = st.columns([1,8])

	with spotify_image_left:
		spotify_logo = st.image("spotify.png")

	#EDA / Data Cleaning
	st.markdown("# EDA / Data Cleaning")

	#EDA / Data Cleaning
	st.markdown("## __Data Cleaning__")


	st.markdown("""
	### Let's first look at our data...

	With a total of 64 gb of user data, we sampled our data using Pyspark and \
	split samples into training and test sets to apply our models.

		""")

	spark_left, spark_right = st.columns(2)

	with spark_left:
		st.image("spotify_streamlit_photos/pyspark_screenshot.jpg")

	with spark_right:
		st.write("This is code for how we came up with the samples in pyspark.")
		st.write("1) import all required packages")
		st.write("2) create a new spark session")
		st.write("3) load the dataframe so we can sample it")
		st.write("4) convert to work with pandas")

	#importing data and combining into one dataframe
	st.markdown("### Shown below is the sampled dataset...")

	log_data = pd.read_csv("data/training_set/log_mini.csv")
	track_data = pd.read_csv("data/track_features/tf_mini.csv")
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
		st.image("spotify_streamlit_photos/skip_eda.png")
	with skip_right:
		st.write("In our sample dataset, there are 111996 skipped entries and 55884 not_skipped entries.")

	st.markdown("#### pause_before_play vs. Skip Behavior")
	pause_left, pause_right = st.columns(2)
	with pause_left:
		st.image("spotify_streamlit_photos/pause_eda.png")
		st.caption("True percentage of users not skipping the current song, grouped by how long of a pause \
		the user takes before playing the current track.")
	with pause_right:
		st.image("spotify_streamlit_photos/pause_eda_plot.png")
		st.caption("Boxplot showing the number of users who skipped and not skipped grouped by how long of a pause \
		the user takes before playing the current track.")

	st.markdown("#### Premium vs. Skip Behavior")
	premium_left, premium_right = st.columns(2)
	with premium_left:
		st.image("spotify_streamlit_photos/premium_eda.png")
		st.caption("True percentage of users not skipping the current song, grouped by whether the user is premium or not.")
	with premium_right:
		st.image("spotify_streamlit_photos/premium_eda_plot.png")
		st.caption("Boxplot showing the number of users who skipped and not skipped grouped by whether the user is premium or not.")

	st.markdown("#### Shuffle vs. Skip Behavior")
	shuffle_left, shuffle_right = st.columns(2)
	with shuffle_left:
		st.image("spotify_streamlit_photos/shuffle_eda.png")
		st.caption("True percentage of users not skipping the current song, grouped by whether the user is in shuffle mode.")
	with shuffle_right:
		st.image("spotify_streamlit_photos/shuffle_eda_plot.png")
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
		st.image("spotify_streamlit_photos/danceability_boxplot.jpg")
		st.caption("Boxplot showing the relationship between a track's danceability and users' skip behavior.")

	with duration_right:
		st.image("spotify_streamlit_photos/duration_boxplot.jpg")
		st.caption("Boxplot showing the relationship between a track's duration and users' skip behavior")

	bar_leftspacer, music_bar_left, music_bar, music_bar_right, bar_rightspacer = st.columns([10,1.5,1.5,1.5,10])

	with music_bar:
		play_button = st.image("spotify_streamlit_photos/spotify_play_button.png")
		# if play_button:
		# 	play_button = st.image("pause_button.png")
	with music_bar_right:
		st.image("spotify_streamlit_photos/skip_button_spotify.png", use_column_width = True)
	st.progress(20)




def model(prev_vars):
	spotify_image_left, spotify_image_right = st.columns([1,8])

	with spotify_image_left:
	  spotify_logo = st.image("spotify.png")


	st.markdown('# Spotify Behavior Model')

	#BASELINE sklearn model, FEEL FREE TO EDIT
	log_data = pd.read_csv("data/training_set/log_mini.csv")
	track_data = pd.read_csv("data/track_features/tf_mini.csv")
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
	    st.image('spotify_streamlit_photos/model_code.jpg')


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
	  play_button = st.image("spotify_streamlit_photos/spotify_play_button.png")
	  # if play_button:
	  #   play_button = st.image("pause_button.png")
	with music_bar_right:
	  st.image("spotify_streamlit_photos/skip_button_spotify.png", use_column_width = True)
	st.progress(40)



def discussion(prev_vars): #problems/problems
	spotify_image_left, spotify_image_right = st.columns([1,8])

	with spotify_image_left:
		spotify_logo = st.image("spotify.png")

	#Title of Page
	st.markdown("# Discussion of Methods")


	#problems with sklearn
	st.markdown("""## __Experiences with scikit-learn__	""")

	st.image("spotify_streamlit_photos/sklearn.png", width = 1000)

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
	st.markdown("""## __Problems with the Behavior Class__""")

	st.image("spotify_streamlit_photos/clustering.png")

	st.markdown("""
	The behavior class is the prediction model we created from scratch; we had much more
	freedom to create our own models based on the specifications of what we wanted to do.
	The main aspect we wanted to change from our sklearn model, was the fact that we
	weren't able to group users' listening data together while predicting if a user
	would skip a song. The prediction was generalized to all users in the dataset.

	However, we still came across some issues with our behavior class, especially since
	we had no starting framework and there were too many elements to account for that
	were much more easily done using the sklearn model, which is eventually what
	we used for our final model.

		""")

	st.markdown('#')
	st.markdown('#')


	bar_leftspacer, music_bar_left, music_bar, music_bar_right, bar_rightspacer = st.columns([10,1.5,1.5,1.5,10])

	with music_bar:
		play_button = st.image("spotify_streamlit_photos/spotify_play_button.png")
		# if play_button:
		# 	play_button = st.image("pause_button.png")
	with music_bar_right:
		st.image("spotify_streamlit_photos/skip_button_spotify.png", use_column_width = True)
	st.progress(65)



def about_us(prev_vars): #About Us Page
	spotify_image_left, spotify_image_right = st.columns([1,8])

	with spotify_image_left:
		spotify_logo = st.image("spotify.png")

	#Title of Page
	st.markdown("""## __About Our Team__

	### Hello! We are undergraduate data science students at the University of California - San Diego
		""")


	images = ["spotify_streamlit_photos/brian.jpg", "spotify_streamlit_photos/annie.jpg", "spotify_streamlit_photos/victor.jpg",
	"spotify_streamlit_photos/aishani.jpg"]
	#st.image(images, width=300, caption=["Brian H", "Annie Fan", "Victor Thai", "Aishani Mohapatra" ])
	brian, annie, victor, aishani = st.columns(4)
	with brian:
		brian = st.image(images[0])
		st.markdown("#### Brian Huang")
		st.write("My name is Brian and I was born June 28 in Singapore. I am currently a Data science major and psychology minor at Warren College. \
		I aspire to apply my data analysis skills to biology or medicine to speed up research processes. I have two sisters, Katherine and Angelina, and I enjoy playing tennis and cycling.")
	with annie:
		annie = st.image(images[1])
		st.markdown('#### Annie Fan')
		st.write("My name is Annie Fan and I am a third year student majoring in data science and cognitive science. \
			I grew up in China and I attended high school in Seattle, Washington. I am into data analysis and machine learning. \
			Currently, I am learning data management with SQL and building recommender systems with data mining.")
	with victor:
		victor = st.image(images[2])
		st.markdown('#### Victor Thai')
		st.write("My name is Victor and I am a second year studying Data Science at UCSD. I grew up in Oakland, CA and have just recently moved to San Diego to pursue my studies. \
			In my free time, I enjoy staying active and being in nature by going on hikes. I plan on using data science to create improvements to heatlthcare technologies and procedures.")
	with aishani:
		aishani = st.image(images[3])
		st.markdown('#### Aishani Mohapatra')
		st.write("My name is Aishani and I am a second year at Muir College studying data science from San Ramon, CA. I am interested in exploring topics the intersection between language and machine learning, \
			 particularly with Natural Language Processing. In my free time, she enjoys singing and hiking with friends.")
	#st.image(images, use_column_width = False, caption=["Brian", "Annie", "Victor", "Aishani" ])

	st.markdown('#')
	st.markdown('#')

	bar_leftspacer, music_bar_left, music_bar, music_bar_right, bar_rightspacer = st.columns([10,1.5,1.5,1.5,10])

	with music_bar:
		play_button = st.image("spotify_streamlit_photos/spotify_play_button.png")
		# if play_button:
		# 	play_button = st.image("pause_button.png")
	with music_bar_right:
		st.image("spotify_streamlit_photos/skip_button_spotify.png", use_column_width = True)
	st.progress(90)




app.set_initial_page(startpage) #home/landing page
app.add_app("EDA", eda) #Adds second page (eda) to the framework
app.add_app("Model", model) #Adds third page (model) to the framework
app.add_app("Discussion", discussion) #Adds fourth page (discussion) to the framework
app.add_app("About Us", about_us) #Adds last page (about us) to the framework

app.run() #Runs the multipage app!
