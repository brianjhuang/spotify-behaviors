import streamlit as st
import pandas as pd
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

#default page setup
st.set_page_config(layout='wide')
spotify_image_left, spotify_image_right = st.columns([1,8])

with spotify_image_left:
  spotify_logo = st.image("spotify.png")


st.markdown('# Spotify Behavior Model')

#BASELINE sklearn model, FEEL FREE TO EDIT
log_data = pd.read_csv("data/training_set/log_mini.csv")
track_data = pd.read_csv("data/track_features/tf_mini.csv")
log_data = log_data.rename(columns = {'track_id_clean':'track_id'})


st.markdown("## __Sklearn Model__ ")
st.write("Using Sklearn, a machine learning package used alongside python, we implemented \
  Logistic Regression and Random Forest Classifier techniques to predict skip behavior \
  given specific musical tracks.")
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

predict_col1, predict_col2, predict_col3, predict_col4 = st.beta_columns(4)

with predict_col1:
  predict_button = st.button("Predict")
with predict_col2:
  st.write("< interact with this button!")

if predict_button:

  sk_col1, sk_col2, sk_col3, sk_col4 = st.beta_columns(4)

  with sk_col1:
    st.write(predictions)
  with sk_col2:
    score = pl.score(x_test, y_test)
    st.write(score)
    st.write("The prediction accuracy score is " + str(score) + "!")