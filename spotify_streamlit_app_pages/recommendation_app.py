import streamlit as st
from multipage_template import save, MultiPage, start_app, clear_cache
import pandas as pd
from PIL import Image
import os
from pyspark.sql import functions as f
from pyspark.sql import SparkSession
import random
from sklearn.decomposition import PCA
import math
from spotifyAPI import Spotify

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

# track_uri = '0gplL1WMoJ6iYaPgMCL0gX'
# st.components.v1.html('<iframe src="https://open.spotify.com/embed/track/' + track_uri + '" width="400" height="100%" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>', width=None, height=None, scrolling=False)

#create spark session
# spark = SparkSession.builder.getOrCreate()
# spark_fp = os.path.join("../sampled_users_100000.csv")
# df = spark.read.option("header", "true").csv(spark_fp)
# users = df.toPandas()
# tf_path_one = os.path.join("/", "Users", "victhaaa", "Spotify", "tf_000000000000.csv")
# tf_path_two = os.path.join("/", "users", "victhaaa", "Spotify", "tf_000000000001.csv")

# track_features_one = pd.read_csv(tf_path_one)
# track_features_two = pd.read_csv(tf_path_two)

# track_features = pd.concat([track_features_one, track_features_two])

# userFeatures = pd.merge(users, track_features, left_on = 'track_id_clean', right_on = 'track_id')
# nonModifiedFeatures = pd.merge(users, track_features, left_on = 'track_id_clean', right_on = 'track_id')

# cols = list(userFeatures.columns)
# drop = cols[1:25]
# userFeatures.drop(columns = drop, inplace = True)
# userFeatures['mode'] = userFeatures['mode'].apply(lambda x: 1 if x == 'major' else 0)

# features = userFeatures.groupby('session_id').mean()
# X = features.reset_index().drop('session_id', axis = 1)
# #we wanted to groupby so we would cluster by user avg song features

# from sklearn.cluster import KMeans
# cluster = KMeans(n_clusters = 3)
# cluster.fit(X)

# y = userFeatures.groupby('session_id').mean()
# #we clustered on session id, so y lets us add the labels by user
# y['Cluster'] = cluster.labels_

# userOne = y[y['Cluster'] == 0]
# userTwo = y[y['Cluster'] == 1]
# userThree = y[y['Cluster'] == 2]


# userOneFeatures = userOne.merge(nonModifiedFeatures, on = 'session_id', how = 'inner')
# userTwoFeatures = userTwo.merge(nonModifiedFeatures, on = 'session_id', how = 'inner')
# userThreeFeatures = userThree.merge(nonModifiedFeatures, on = 'session_id', how = 'inner')

# userOneFeatures['not_skipped'] = userOneFeatures['not_skipped'].apply(lambda x: 1 if x == True else 0)
# userTwoFeatures['not_skipped'] = userTwoFeatures['not_skipped'].apply(lambda x: 1 if x == True else 0)
# userThreeFeatures['not_skipped'] = userThreeFeatures['not_skipped'].apply(lambda x: 1 if x == True else 0)

# userOneFeatures['premium']= userOneFeatures['premium'].apply(lambda x: 1 if x is True else 0)
# userOneFeatures['hist_user_behavior_is_shuffle'] = userOneFeatures['hist_user_behavior_is_shuffle'].apply(lambda x: 1 if x is True else 0)

# userTwoFeatures['premium']= userTwoFeatures['premium'].apply(lambda x: 1 if x is True else 0)
# userTwoFeatures['hist_user_behavior_is_shuffle'] = userTwoFeatures['hist_user_behavior_is_shuffle'].apply(lambda x: 1 if x is True else 0)

# userThreeFeatures['premium']= userThreeFeatures['premium'].apply(lambda x: 1 if x is True else 0)
# userThreeFeatures['hist_user_behavior_is_shuffle'] = userThreeFeatures['hist_user_behavior_is_shuffle'].apply(lambda x: 1 if x is True else 0)

# userOneFeatures.drop(['acousticness_x', 'beat_strength_x', 'bounciness_x',
#        'danceability_x', 'dyn_range_mean_x', 'energy_x', 'flatness_x',
#        'instrumentalness_x', 'key_x', 'liveness_x', 'loudness_x',
#        'mechanism_x', 'mode_x', 'organism_x', 'speechiness_x', 'tempo_x',
#        'time_signature_x', 'valence_x', 'acoustic_vector_0_x',
#        'acoustic_vector_1_x', 'acoustic_vector_2_x', 'acoustic_vector_3_x',
#        'acoustic_vector_4_x', 'acoustic_vector_5_x', 'acoustic_vector_6_x',
#        'acoustic_vector_7_x', 'Cluster', 'session_position', 'session_length', 'not_skipped', 'context_switch',
#        'no_pause_before_play', 'short_pause_before_play',
#        'long_pause_before_play', 'hist_user_behavior_n_seekfwd',
#        'hist_user_behavior_n_seekback', 'hist_user_behavior_is_shuffle',
#        'hour_of_day', 'premium', 'context_type',
#        'hist_user_behavior_reason_start', 'duration', 'release_year',
#        'us_popularity_estimate'], axis = 1, inplace = True)
# userTwoFeatures.drop(['acousticness_x', 'beat_strength_x', 'bounciness_x',
#        'danceability_x', 'dyn_range_mean_x', 'energy_x', 'flatness_x',
#        'instrumentalness_x', 'key_x', 'liveness_x', 'loudness_x',
#        'mechanism_x', 'mode_x', 'organism_x', 'speechiness_x', 'tempo_x',
#        'time_signature_x', 'valence_x', 'acoustic_vector_0_x',
#        'acoustic_vector_1_x', 'acoustic_vector_2_x', 'acoustic_vector_3_x',
#        'acoustic_vector_4_x', 'acoustic_vector_5_x', 'acoustic_vector_6_x',
#        'acoustic_vector_7_x', 'Cluster', 'session_position', 'session_length', 'not_skipped', 'context_switch',
#        'no_pause_before_play', 'short_pause_before_play',
#        'long_pause_before_play', 'hist_user_behavior_n_seekfwd',
#        'hist_user_behavior_n_seekback', 'hist_user_behavior_is_shuffle',
#        'hour_of_day', 'premium', 'context_type',
#        'hist_user_behavior_reason_start', 'duration', 'release_year',
#        'us_popularity_estimate'], axis = 1, inplace = True)
# userThreeFeatures.drop(['acousticness_x', 'beat_strength_x', 'bounciness_x',
#        'danceability_x', 'dyn_range_mean_x', 'energy_x', 'flatness_x',
#        'instrumentalness_x', 'key_x', 'liveness_x', 'loudness_x',
#        'mechanism_x', 'mode_x', 'organism_x', 'speechiness_x', 'tempo_x',
#        'time_signature_x', 'valence_x', 'acoustic_vector_0_x',
#        'acoustic_vector_1_x', 'acoustic_vector_2_x', 'acoustic_vector_3_x',
#        'acoustic_vector_4_x', 'acoustic_vector_5_x', 'acoustic_vector_6_x',
#        'acoustic_vector_7_x', 'Cluster', 'session_position', 'session_length', 'not_skipped', 'context_switch',
#        'no_pause_before_play', 'short_pause_before_play',
#        'long_pause_before_play', 'hist_user_behavior_n_seekfwd',
#        'hist_user_behavior_n_seekback', 'hist_user_behavior_is_shuffle',
#        'hour_of_day', 'premium', 'context_type',
#        'hist_user_behavior_reason_start', 'duration', 'release_year',
#        'us_popularity_estimate'], axis = 1, inplace = True)


# userOneFeatures.rename(columns = lambda x: x[:-2] if x[-2:] == '_y' else x, inplace = True)
# userTwoFeatures.rename(columns = lambda x: x[:-2] if x[-2:] == '_y' else x, inplace = True)
# userThreeFeatures.rename(columns = lambda x: x[:-2] if x[-2:] == '_y' else x, inplace = True)

# userOneFeatures.drop(['track_id_clean', 
#          'skip_1', 
#          'skip_2', 
#          'skip_3',
#          'hist_user_behavior_reason_end',
#          'track_id',
#          'date'], 
#         axis = 1, inplace = True)

# userTwoFeatures.drop(['track_id_clean', 
#          'skip_1', 
#          'skip_2', 
#          'skip_3',
#          'hist_user_behavior_reason_end',
#          'track_id',
#          'date'], 
#         axis = 1, inplace = True)

# userThreeFeatures.drop(['track_id_clean', 
#          'skip_1', 
#          'skip_2', 
#          'skip_3',
#          'hist_user_behavior_reason_end',
#          'track_id',
#          'date'], 
#         axis = 1, inplace = True)

# userOneFeatures.drop(['acoustic_vector_0',
#  'acoustic_vector_1',
#  'acoustic_vector_2',
#  'acoustic_vector_3',
#  'acoustic_vector_4',
#  'acoustic_vector_5',
#  'acoustic_vector_6',
#  'acoustic_vector_7',
#  'beat_strength',
#  'bounciness',
#  'dyn_range_mean',
#  'flatness',
#  'mechanism',
#  'organism'], axis = 1, inplace = True)

# userTwoFeatures.drop(['acoustic_vector_0',
#  'acoustic_vector_1',
#  'acoustic_vector_2',
#  'acoustic_vector_3',
#  'acoustic_vector_4',
#  'acoustic_vector_5',
#  'acoustic_vector_6',
#  'acoustic_vector_7',
#  'beat_strength',
#  'bounciness',
#  'dyn_range_mean',
#  'flatness',
#  'mechanism',
#  'organism'], axis = 1, inplace = True)

# userThreeFeatures.drop(['acoustic_vector_0',
#  'acoustic_vector_1',
#  'acoustic_vector_2',
#  'acoustic_vector_3',
#  'acoustic_vector_4',
#  'acoustic_vector_5',
#  'acoustic_vector_6',
#  'acoustic_vector_7',
#  'beat_strength',
#  'bounciness',
#  'dyn_range_mean',
#  'flatness',
#  'mechanism',
#  'organism'], axis = 1, inplace = True)

# userOneFeatures['mode'] = userOneFeatures['mode'].apply(lambda x: 1 if x == 'major' else 0)
# userTwoFeatures['mode'] = userTwoFeatures['mode'].apply(lambda x: 1 if x == 'major' else 0)
# userThreeFeatures['mode'] = userThreeFeatures['mode'].apply(lambda x: 1 if x == 'major' else 0)

userOneFeatures = pd.read_csv("../userOneFeatures.csv")
userTwoFeatures = pd.read_csv("../userTwoFeatures.csv")
userThreeFeatures = pd.read_csv("../userThreeFeatures.csv")

userOneFeatures.drop('Unnamed: 0', axis = 1, inplace = True)
userTwoFeatures.drop('Unnamed: 0', axis = 1, inplace = True)
userThreeFeatures.drop('Unnamed: 0', axis = 1, inplace = True)


from spotifyAPI import Spotify

s = Spotify()
st.write('Authentication Passed')

def selectSong(song):
	return modelOutput[song]

st.write('Getting the Top 50 - USA...')

# songs = s.get_playlist_items()

# songIndex = {}
# for i in range(len(songs)):
# 	songIndex[songs[i]] = i

# dropDown = st.selectbox('What song would you like to pick?',
# 	(tuple(songs)))

features = s.get_playlist_features('Top 50 - USA')
st.write('Running Model...')


from songRecommender import songRecommender

model = songRecommender(data = userTwoFeatures, predict = features)

modelOutput = model.similar(model.getPredict(), model.getFeatures())

st.write(modelOutput)

# songOutput = selectSong(option)

# st.write(selectSong(option))
# track_uri = str(songOutput.keys()[0])
# st.components.v1.html('<iframe src="https://open.spotify.com/embed/track/' + track_uri + '" width="400" height="100%" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>', width=None, height=None, scrolling=False)









# track_uri = ''

# def player_func(features):
# 	track_uri = features['id']


# song_url = st.components.v1.html('<iframe src="https://open.spotify.com/embed/album/' + track_uri + '" width="300" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>', width=None, height=None, scrolling=False)
