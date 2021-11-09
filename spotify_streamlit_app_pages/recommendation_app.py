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

#create spark session
spark = SparkSession.builder.getOrCreate()
spark_fp = os.path.join("/","Users","victhaaa","spotify-behaviors","sampled_users.csv")
df = spark.read.option("header", "true").csv(spark_fp)
users = df.toPandas()
tf_path_one = os.path.join("/", "Users", "victhaaa", "spotify-behaviors", "data", "track_features", "tf_mini.csv")
tf_path_two = os.path.join("/", "users", "victhaaa", "spotify-behaviors", "data", "track_features", "tf_mini.csv")

track_features_one = pd.read_csv(tf_path_one)
track_features_two = pd.read_csv(tf_path_two)
track_features = pd.concat([track_features_one, track_features_two])

userFeatures = pd.merge(users, track_features, left_on = 'track_id_clean', right_on = 'track_id')
nonModifiedFeatures = pd.merge(users, track_features, left_on = 'track_id_clean', right_on = 'track_id')

cols = list(userFeatures.columns)

drop = cols[1:25]
userFeatures.drop(columns = drop, inplace = True)
userFeatures['mode'] = userFeatures['mode'].apply(lambda x: 1 if x == 'major' else 0)

features = userFeatures.groupby('session_id').mean()

X = features.reset_index().drop('session_id', axis = 1)
#we wanted to groupby so we would cluster by user avg song features

from sklearn.cluster import KMeans
cluster = KMeans(n_clusters = 3)
cluster.fit(X)

y = userFeatures.groupby('session_id').mean()

y['Cluster'] = cluster.labels_

userOne = y[y['Cluster'] == 0]

userOneFeatures = userOne.merge(nonModifiedFeatures, on = 'session_id', how = 'inner')

userOneFeatures['not_skipped'] = userOneFeatures['not_skipped'].apply(lambda x: 1 if x == True else 0)

userOneFeatures['premium']= userOneFeatures['premium'].apply(lambda x: 1 if x is True else 0)
userOneFeatures['hist_user_behavior_is_shuffle'] = userOneFeatures['hist_user_behavior_is_shuffle'].apply(lambda x: 1 if x is True else 0)

userOneFeatures.drop(['acousticness_x', 'beat_strength_x', 'bounciness_x',
       'danceability_x', 'dyn_range_mean_x', 'energy_x', 'flatness_x',
       'instrumentalness_x', 'key_x', 'liveness_x', 'loudness_x',
       'mechanism_x', 'mode_x', 'organism_x', 'speechiness_x', 'tempo_x',
       'time_signature_x', 'valence_x', 'acoustic_vector_0_x',
       'acoustic_vector_1_x', 'acoustic_vector_2_x', 'acoustic_vector_3_x',
       'acoustic_vector_4_x', 'acoustic_vector_5_x', 'acoustic_vector_6_x',
       'acoustic_vector_7_x', 'Cluster', 'session_position', 'session_length', 'not_skipped', 'context_switch',
       'no_pause_before_play', 'short_pause_before_play',
       'long_pause_before_play', 'hist_user_behavior_n_seekfwd',
       'hist_user_behavior_n_seekback', 'hist_user_behavior_is_shuffle',
       'hour_of_day', 'premium', 'context_type',
       'hist_user_behavior_reason_start', 'duration', 'release_year',
       'us_popularity_estimate'], axis = 1, inplace = True)

userOneFeatures.rename(columns = lambda x: x[:-2] if x[-2:] == '_y' else x, inplace = True)

userOneFeatures.drop(['track_id_clean', 
         'skip_1', 
         'skip_2', 
         'skip_3',
         'hist_user_behavior_reason_end',
         'track_id',
         'date'], 
        axis = 1, inplace = True)

userOneFeatures['mode'] = userOneFeatures['mode'].apply(lambda x: 1 if x == 'major' else 0)

class songRecommender():
    data = {}
    features = []
    predictFeatures = []

    def __init__(self, data, predict):
        '''
        data - our persona user's information
        predict - the new songs from the API
        '''
        
        
        self.data = self.parseData(data)
        #parse the new data
        self.features = self.featureVector(self.data) #apply PCA
        #generate features for the new data
        self.predictFeatures = self.featureAPIVector(predict) #apply PCA
        #clean the api data

    def parseData(self, data):

        import json

        parsed = json.loads(data.to_json(orient = 'records'))
        cleaned = {}

        for line in parsed:


            featuresSet = ['acousticness', 'beat_strength', 'bounciness', 'danceability',
               'dyn_range_mean', 'energy', 'flatness', 'instrumentalness', 'key',
               'liveness', 'loudness', 'mechanism', 'mode', 'organism', 'speechiness',
               'tempo', 'time_signature', 'valence', 'acoustic_vector_0',
               'acoustic_vector_1', 'acoustic_vector_2', 'acoustic_vector_3',
               'acoustic_vector_4', 'acoustic_vector_5', 'acoustic_vector_6',
               'acoustic_vector_7']
            #get only user behaviors

            featuresDict = {k:v for k,v in line.items() if k in featuresSet}
            cleaned[line['session_id']] = featuresDict

        return cleaned

    def featureVector(self, data):
        #transform our dictionary of song features into a matrix of feature vectors
        vector = []

        for k in data:
            d = dict(sorted(data[k].items()))
            vector.append((k, d))

        return vector

    def featureAPIVector(self, data):
        #transform our API features into usable data
        vector = []
        keep = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
        for d in data:
            temp = {k:v for k, v in d.items() if k in keep}
            temp = dict(sorted(temp.items()))
            vector.append((d['uri'],temp))

        return vector

    def getData(self):
        return self.data
    
    def getFeatures(self):
        return self.features
    
    def getPredict(self):
        return self.predictFeatures

    def cosine(self, feature, features, N):
        '''
        feature - a feature vector of tuples, with index 0 being link and 1 being the vector
        feature is the song from the API
        features - all feature vectors belonging to current persona user
        all the songs in our generated user (data)
        N - number of similiar songs we want to return
        '''
        similarities = []

        numer = 0
        denom1 = 0
        denom2 = 0

        for featureTwo in features:
            sim = 0
            numer = sum([a * b for a, b in zip(list(feature[1].values()), list(featureTwo[1].values()))])
            denom1 = sum([l ** 2 for l in list(feature[1].values())])
            denom2 = sum([l ** 2 for l in list(featureTwo[1].values())])
            denom = math.sqrt(denom1) * math.sqrt(denom2)
            if denom == 0:
                sim = 0
            sim = numer/denom

            similarities.append((sim, featureTwo[0]))

        similarities.sort(reverse = True)
        return similarities[:N]
    
    def similar(self, X, y):
        predictions = []
        for feature in X:
            entry = {feature[0]:cosine(feature, y, 1)[0]}
            #figure out why it keeps returning 10 entries
            predictions.append(entry)
        return predictions

model = songRecommender(data = userOneFeatures, predict = features)

model.similar(model.getPredict(), model.getFeatures())[:5]

st.write(features)
st.write(userOneFeatures)


track_uri = ''

def player_func(features):
	track_uri = features['id']


song_url = st.components.v1.html('<iframe src="https://open.spotify.com/embed/album/' + track_uri + '" width="300" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>', width=None, height=None, scrolling=False)
