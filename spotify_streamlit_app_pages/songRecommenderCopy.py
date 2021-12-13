import math
from spotifyAPI import Spotify

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd
import json

class songRecommender():
    '''
    Our song recommender class. 
    Utlizes the Spotify API to gather data.
    Preprocesses our data and standardizes song features.
    Generates recommendations using cosine-similarity.
    
    Parameters:
    data (dictionary) - all the data we are using
    features (list) - all the features we are predicitng with
    predictFeatures (list) - all the features we get using the Spotify API.
    '''

    data = {}
    features = []
    predictFeatures = []

    def __init__(self, data, predict):
        '''
        Our constructor. Gets and cleans our data. 
        Generates a feature vector for both the features we have 
        and the features we have from the Spotify API.
        
        Scales all of our features to the same scale.
        
        Params:
        data (dictionary) - our persona user's information
        predict (dictionary) - the new songs from the API
        '''
        
        self.data = self.parseData(self.dataPreprocessing(data))
        #parse the new data
        self.features = self.featureVector(self.data) 
        #generate features for the new data

        self.predictFeatures = self.featureAPIVector(predict)
        self.predictFeatures = self.scaleAPI(self.getPredict())
        #clean the api data
        
    def dataPreprocessing(self, data):
        '''
        This preprocesses our data for us. Standard Scales all of our data.
        
        Params:
        
        data (dictionary) - the data we want to scale
        
        Returns:
        processed (DataFrame) - the processed data
        '''
        
        cols = data.columns
        
        ss = ['acousticness', 'danceability', 'energy',
       'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
       'speechiness', 'tempo', 'time_signature', 'valence']
        
        as_is = ['session_id']
        
        preproc = ColumnTransformer(
            transformers = [
                ('as_is', FunctionTransformer(lambda x: x), as_is),
                ('standard_scale', StandardScaler(), ss),
            ]
        )
        
        processed = pd.DataFrame(preproc.fit_transform(data), columns = cols)
        return processed
    
    def scaleAPI(self, data):
        '''
        This preprocesses our data for us. Standard Scales all of our data.
        
        Params:
        
        data (dictionary) - the data we want to scale
        
        Returns:
        processed (DataFrame) - the processed data
        '''
        p = data

        ss = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'key',
           'liveness', 'loudness', 'mode', 'speechiness', 'tempo',
           'time_signature', 'valence']

        preproc = ColumnTransformer(
            transformers = [
                ('standard_scale', StandardScaler(), ss)
            ]
        )
        #clean up our API data

        df = pd.DataFrame()
        for entry in p:
            temp = pd.DataFrame.from_dict(data = entry[1], orient = 'index').T
            df = pd.concat([temp, df])
        transformed = pd.DataFrame(preproc.fit_transform(df), columns = ss).to_dict(orient = 'records')
        transformedPredict = []

        for i in range(len(p)):
            transformedPredict.append((p[i][0], transformed[i])) 

        return transformedPredict

    def parseData(self, data):
        '''
        Parse our data and turn it into a dictionary that we can use.
        
        Params:
        data (json) - the data that we want to parse
        
        Returns:
        cleaned (dictionary) - the cleaned data with session_id as the key and the features as the values
        '''

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
            #loop through and get all the features in our dictionary
            cleaned[line['session_id']] = featuresDict

        return cleaned

    def featureVector(self, data):
        '''
        Transforms our dictionary of song features into a matrix of feature vectors
        
        Params:
        
        data (dictionary) - the dictionary of song features
        
        Returns:
        
        vector(list) - the vector of song features
        '''
        vector = []

        for k in data:
            d = dict(sorted(data[k].items()))
            vector.append((k, d))

        return vector

    def featureAPIVector(self, data):
        '''
        Transforms our dictionary of song features into a matrix of feature vectors
        
        Params:
        
        data (dictionary) - the dictionary of song features
        
        Returns:
        
        vector(list) - the vector of song features
        '''
        vector = []
        keep = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
        
        if isinstance(data, dict):
            #if we want to just compare one song
            temp = {k:v for k, v in data.items() if k in keep}
            temp = dict(sorted(temp.items()))
            vector.append((data['uri'], temp))
            return vector
        
        for d in data:
            temp = {k:v for k, v in d.items() if k in keep}
            temp = dict(sorted(temp.items()))
            vector.append((d['uri'],temp))

        return vector

    def getData(self):
        '''
        Get our data.
        
        Returns:
        data (dicionary)
        '''
        return self.data
    
    def getFeatures(self):
        '''
        Get our features.
        
        Returns:
        features (list)
        '''
        return self.features
    
    def getPredict(self):
        '''
        Get our features.
        
        Returns:
        features (list)
        '''
        return self.predictFeatures

    def cosine(self, feature, features, N):
        '''
        Take in a song (feature) which is a song from our API.
        Take in a group of songs (features) which are songs our persona user/user has listened to.
        Return the N amount of similiar songs from our user that are similiar to the song we inputted.
        
        Params:
        feature - a feature vector of tuples, with index 0 being link and 1 being the vector
        feature is the song from the API
        features - all feature vectors belonging to current persona user
        all the songs in our generated user (data)
        N - number of similiar songs we want to return
        
        Returns:
        
        similarities (list) - a list of all of our similarities
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
            else:
                sim = numer/denom

            similarities.append((sim, featureTwo[0]))

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
        predictions = []
        for feature in X:
            entry = {feature[0]:self.cosine(feature, y, 1)}
            predictions.append(entry)
        return predictions[:10]
