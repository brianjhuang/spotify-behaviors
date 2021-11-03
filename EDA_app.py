import pandas as pd
import streamlit as st
import os

#EDA / Data Cleaning
st.markdown("## __Data Cleaning__")

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

log_data = pd.read_csv("data/training_set/log_mini.csv")
track_data = pd.read_csv("data/track_features/tf_mini.csv")
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
	'skip_1','skip_2', and 'skip_3'. These columns are combined into one column with boolean values that \
	shows whether the user has skipped the song.")

st.write("The column 'hist_user_behavior_reason_end' directly determines users' skip behavior. As a result, \
it is dropped for our model to better predict the outcomes without such giveaways. ")

#get_skip
st.markdown("""## __EDA__
We performed exploratory data analysis to explore the patterns between different features \
and users' skipping behavior""")

st.markdown("#### Overall Skip Behavior")
skip_left, skip_right = st.beta_columns(2)

with skip_left:
	st.image("skip_eda.png")
with skip_right:
	st.write("In our sample dataset, there are 111996 skipped entries and 55884 not_skipped entries.")

st.markdown("#### pause_before_play vs. Skip Behavior")
pause_left, pause_right = st.beta_columns(2)
with pause_left:
	st.image("pause_eda.png")
	st.caption("True percentage of users not skipping the current song, grouped by how long of a pause \
	the user takes before playing the current track.")
with pause_right:
	st.image("pause_eda_plot.png")
	st.caption("Boxplot showing the number of users who skipped and not skipped grouped by how long of a pause \
	the user takes before playing the current track.")

st.markdown("#### Premium vs. Skip Behavior")
premium_left, premium_right = st.beta_columns(2)
with premium_left:
	st.image("premium_eda.png")
	st.caption("True percentage of users not skipping the current song, grouped by whether the user is premium or not.")
with premium_right:
	st.image("premium_eda_plot.png")
	st.caption("Boxplot showing the number of users who skipped and not skipped grouped by whether the user is premium or not.")

st.markdown("#### Shuffle vs. Skip Behavior")
shuffle_left, shuffle_right = st.beta_columns(2)
with shuffle_left:
	st.image("shuffle_eda.png")
	st.caption("True percentage of users not skipping the current song, grouped by whether the user is in shuffle mode.")
with shuffle_right:
	st.image("shuffle_eda_plot.png")
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

duration_left, duration_right = st.beta_columns(2)

with duration_left:
	st.image("danceability_boxplot.jpg")
	st.caption("Boxplot showing the relationship between a track's danceability and users' skip behavior.")

with duration_right:
	st.image("duration_boxplot.jpg")
	st.caption("Boxplot showing the relationship between a track's duration and users' skip behavior")
