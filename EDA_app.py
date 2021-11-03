import pandas as pd
import streamlit as st
import os

st.set_page_config(layout='wide')

#EDA / Data Cleaning
st.markdown("## __EDA / Data Cleaning__")

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

duration_left, duration_right = st.columns(2)

with duration_left:
	st.image("spotify_streamlit_photos/danceability_boxplot.jpg")
	st.caption("Data exploration to analyze skip behavior depending on dacibility.")

with duration_right:
	st.image("spotify_streamlit_photos/duration_boxplot.jpg")
	st.caption("Data exploration of whether the duration of a song plays a part in skip behavior.")
