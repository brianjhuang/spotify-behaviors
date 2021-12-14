import math
from spotifyAPI import Spotify

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd
import json

class songRecommender:
    '''
    Our song recommender class.
    Utlizes the Spotify API to gather data.
    Preprocesses our data and standardizes song features.
    Generates recommendations using cosine-similarity.

    Parameters:
    data (dictionary) - all the data we are using
    X (list) - the general listening behavior (such as the Top 50)
    y (list) - the songs we want to recommend for (our playlist)
    songs (list) - the name of the songs from the API.
    '''

    data = []
    X = []
    y = []
    xID = []
    yID = []
    X_songs = []
    y_songs = []

    def __init__(self, data, predict, X_songs, y_songs):
        '''
        Our constructor. Gets and cleans our data.
        Generates a feature vector for both the features we have
        and the features we have from the Spotify API.

        Scales all of our features to the same scale.

        Params:
        data (list of dictionaries) - our user's listening behavior
        predict (list of dictionaries) - general/someone else's listening behavior (the US Top 50)
        '''
        self.X_songs = X_songs
        self.y_songs = y_songs
        self.xID = [song['id'] for song in data]
        self.yID = [song['id'] for song in predict]
        merged = []
        [merged.append((ui, f, i)) for ui, f, i in self.parseData(data + predict) if (ui, f, i) not in merged]
        #get rid of duplicates
        self.data = self.scaleData(merged)
        self.X, self.y = self.splitData(self.getData())

    def getData(self):
        '''
        Getter for our data

        Returns:
        data (list of dictionaries) - a vector of all our acoustic features
        '''
        return self.data

    def getX(self):
        '''
        Getter for our X features.

        Returns:
        X (list of dictionaries) - a vector of all the song features
        '''
        return self.X

    def getY(self):
        '''
        Getter for our y features.

        Returns:
        y (list of dictionaries) - a vector of all the song features
        '''
        return self.y

    def getXSongs(self):
        '''
        Getter for songs.

        Returns:
        songs (list) - the names of the songs on our playlist
        '''
        return self.X_songs

    def getySongs(self):
        '''
        Getter for songs.

        Returns:
        songs (list) - the names of the songs on our playlist
        '''
        return self.y_songs


    def parseData(self, data):
        '''
        Transforms our dictionary of song features into a matrix of feature vectors

        Params:

        data (dictionary) - the dictionary of song features

        Returns:

        vector(list) - the vector of song features
        '''
        vector = []
        keep = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'duration_ms']
        for d in data:
            temp = {k:v for k, v in d.items() if k in keep}
            temp = dict(sorted(temp.items()))
            vector.append((d['uri'],temp, d['id']))

        return vector

    def scaleData(self, data):
        '''
        This preprocesses our data for us. Standard Scales all of our data.

        Params:

        data (dictionary) - the data we want to scale

        Returns:
        processed (DataFrame) - the processed data
        '''
        ss = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'duration_ms']

        preproc = ColumnTransformer(
            transformers = [
                ('standard_scale', StandardScaler(), ss)
            ]
        )

        df = pd.DataFrame()
        for entry in data:
            temp = pd.DataFrame.from_dict(data = entry[1], orient = 'index').T
            df = pd.concat([temp, df])
        transformed = pd.DataFrame(preproc.fit_transform(df), columns = ss).to_dict(orient = 'records')
        transformedPredict = []

        for i in range(len(data)):
            transformedPredict.append((data[i][2], data[i][0], transformed[i]))

        return transformedPredict

    def splitData(self, data):
        '''
        Split our data into the general listener (X) and our song/playlist
        (y)

        Params:
        data (list) - the list of dictionaries with our data.

        Returns:
        y (list) - what we want to predict (songs in our playlist)
        X (list) - what we are using to predict with (songs in the general playlist)
        '''
        X = []
        y = []
        for song in data:
            if song[0] not in X:
                if song[0] in self.xID and song[0]:
                    X.append(song)
            if song[0] not in y:
                if song[0] in self.yID and song[0]:
                    y.append(song)
        return X, y

    def cosine(self, song, songs, N):
        '''
        Take in a song (feature) which is a song from our API.
        Take in a group of songs (features) which are songs our persona user/user has listened to.
        Return the N amount of similiar songs from our user that are similiar to the song we inputted.

        Params:
        song - a single song that we want to find similiar songs for
        songs - the general list of songs that we can compare to
        N - number of similiar songs we want to return

        Returns:

        similarities (list) - a list of all of our similarities
        '''

        similarities = []

        numer = 0
        denom1 = 0
        denom2 = 0

        for songTwo in songs:
            sim = 0
            numer = sum([featureOne * featureTwo for featureOne, featureTwo in zip(list(song[2].values()), list(songTwo[2].values()))])
            denom1 = sum([feature ** 2 for feature in list(song[2].values())])
            denom2 = sum([feature ** 2 for feature in list (songTwo[2].values())])
            denom = math.sqrt(denom1) * math.sqrt(denom2)
            if denom == 0:
                sim = 0
            else:
                sim = numer/denom
            similarities.append((sim, songTwo[0], songTwo[1]))
        similarities.sort(reverse = True)
        return similarities[:N]

    def similar(self, X, y):
        '''
        Runs cosine similarity on our entire feature vector.

        Params:
        X (dictionary) - our feature matrix
        y (list) - the items we want to compare to

        Returns:

        predictions (list) - our predictions
        '''
        predictions = {}
        songID = 0
        for feature in y:
            if songID == 101:
                break
            entry = (feature[1],self.cosine(feature, X, 5))
            #figure out why it keeps returning 10 entries
            predictions[self.getySongs()[songID]] = entry
            songID += 1
        return predictions
