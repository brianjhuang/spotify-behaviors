import math
from spotifyAPI import Spotify

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd

class songRecommender():

    data = {}
    features = []
    predictFeatures = []

    def __init__(self, data, predict):
        '''
        data - our persona user's information
        predict - the new songs from the API
        '''
        
        self.data = self.parseData(self.dataPreprocessing(data))
        #parse the new data
        self.features = self.featureVector(self.data) 
        #generate features for the new data
        self.predictFeatures = self.featureAPIVector(predict)
        self.predictFeatures = self.scaleAPI(self.getPredict())
        #clean the api data
        
    def dataPreprocessing(self, data):
        
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
        p = data

        ss = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'key',
           'liveness', 'loudness', 'mode', 'speechiness', 'tempo',
           'time_signature', 'valence']

        preproc = ColumnTransformer(
            transformers = [
                ('standard_scale', StandardScaler(), ss)
            ]
        )

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
            entry = {feature[0]:self.cosine(feature, y, 1)}
            #figure out why it keeps returning 10 entries
            predictions.append(entry)
        return predictions
