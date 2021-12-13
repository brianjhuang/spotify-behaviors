import pandas as pd
import streamlit as st
import os


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# creates a wider page on streamlit
st.set_page_config(layout="wide")

# Background image of website (dark grey)
st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://www.fredrogerscenter.org/wp-content/uploads/2015/11/dark-grey-background-FRC-Grey.png")
    }
   .sidebar .sidebar-content {
        background: url("https://upload.wikimedia.org/wikipedia/commons/5/50/Black_colour.jpg")
    }
    </style>
    """,
    unsafe_allow_html=True
)

#grey image
#https://www.fredrogerscenter.org/wp-content/uploads/2015/11/dark-grey-background-FRC-Grey.png


#Title of website
st.markdown("# Spotify Listener Skipping Behavior")

#Author names
st.markdown(
	"""
Created by Annie Fan, Brian Huang, Aishani Mohapatra, Victor Thai

""")

# #Sidebar 
# st.sidebar.button("Spotify Home",)

#first columns
intro_left, intro_right = st.beta_columns(2)

with intro_left:
	st.image("spotify_cover_image.jpg")
	st.caption("Spotify app displayed on smartphone.  " +
	"https://www.businessinsider.com/how-to-see-spotify-listening-history")

with intro_right:
	st.markdown("#  Will You Skip The Next Song?")
	st.write("This project focues on analyzing the recommender systems among \
		Spotify's application, a popular music streaming service. To determine\
		 how Spotify is able to recommend songs to their listeners, we will \
		 look at specific musical traits among tracks that listeners \
		 have listened to, in order to predict whether they will play \
		 through or skip a particular song.")

#intro to dataset, process of getting
st.write("intro to dataset, process of getting, 400 gb")


#EDA / Data Cleaning
st.markdown("## __EDA / Data Cleaning__")

st.markdown("""
### Let's first look at our data...

With a total of 64 gb of user data, we sampled our data using Pyspark and \
split samples into training and test sets to apply our models. 

	""")

spark_left, spark_right = st.beta_columns(2)

with spark_left:
	st.image("pyspark_screenshot.jpg")

with spark_right:
	st.write("This is code for how we came up with the samples in pyspark.")
	st.write("1) import all required packages")
	st.write("2) create a new spark session")
	st.write("3) load the dataframe so we can sample it")
	st.write("4) convert to work with pandas")
	 
#importing data and combining into one dataframe
st.markdown("### Shown below is the sampled dataset...")

log_data = pd.read_csv("Spotify_Data/data/training_set/log_mini.csv")
track_data = pd.read_csv("Spotify_Data/data/track_features/tf_mini.csv")
log_data = log_data.rename(columns = {'track_id_clean':'track_id'})

df = pd.merge(log_data,track_data,on='track_id',how='left')

#display the dataframe
st.write(df.head(100))

#display the columns of track and log data
log_left, track_right = st.beta_columns(2)

with log_left:
	st.markdown("### Log Data Columns")
	st.write(log_data.columns)

with track_right:
	st.markdown("### Track Data Columns")
	st.write(track_data.columns)

link = st.beta_expander("To better understand the meaning of each column, follow this link.")
link.write('https://drive.google.com/file/d/1aR6g0hGhue3pGZ81buEXvRCFLmdMfFep/view?usp=sharing')


st.write("Among the Log Data, skip results are given by the columns 'not_skipped', \
	'skip_1','skip_2', and 'skip_3'. As a result these, columns must be dropped for \
	our model to better predict the outcomes without such giveaways.")

st.write("The columns 'skip_1', 'skip_2', 'skip_3', and 'not_skipped' are dropped as they give skip behavior away.")
st.write("These are the updated columns that will be used for our model.")

#get_skip
st.markdown("""
### Analyzing duration of song with skips
	""")
def get_skip(df):
    if df['not_skipped'] == 1:
        return 'not skipped'
    else: 
        return 'skipped'
skip_info = df.apply(get_skip, axis = 1)
df = df.assign(skip_type = skip_info)

duration_left, duration_right = st.beta_columns(2)

with duration_left:
	st.image("danceability_boxplot.jpg")
	st.caption("Data exploration to analyze skip behavior depending on dacibility.")

with duration_right:
	st.image("duration_boxplot.jpg")
	st.caption("Data exploration of whether the duration of a song plays a part in skip behavior.")


#sklearn model
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


#problems with sklearn
st.markdown("""## __Problems with Sklearn__

Using sklearn to generate a model that would predict skipping behavior allowed for 
data from different users to become a general prediction. In the real world, this
is not accurate as every listener has their own unique taste and skipping beahvior.
In other words, we simply cannot assume that everyone shares the same behavior through
sampling from the whole dataset.

Rather, it must be recognized that each listener has their unique listening styles.
Therefore, it is essential that we sample the listening behavior of each specific listener
in order to determine traits that shape the skipping behavior of listeners.
	""")

#behavior class
st.markdown(""" ## __Our Behavior Class__

Since realizing that skipping behavior varies among all listeners, we pioneered
our own listening behavior class model. This model takes into consideration the 
variable 'user_id', also known as the unique user's identification number.
	""")


#About us
st.markdown("""## __About Our Team__

### Hello! We are undergraduate students at the University of California - San Diego
	""")
