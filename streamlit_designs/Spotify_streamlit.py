import pandas as pd
import streamlit as st
import webbrowser

# from pyspark.sql import functions as f
# from pyspark.sql import SparkSession
# import os

# #spark session and data loading
# spark = SparkSession.builder.getOrCreate()

# spark_fp = os.path.join("..", "Spotify", "track_features_subset_0.csv")
# # doing this temporarily so i can show graphs
# # will have to figure out a way to display graphs without loading in data like this...

# df = spark.read.load(spark_fp, 
#                       format="csv", inferSchema="true", header="true")


# Background image of website (dark grey)
# st.markdown(
#     """
#     <style>
#     .reportview-container {
#         background: url("https://www.fredrogerscenter.org/wp-content/uploads/2015/11/dark-grey-background-FRC-Grey.png")
#     }
#    .sidebar .sidebar-content {
#         background: url("https://upload.wikimedia.org/wikipedia/commons/5/50/Black_colour.jpg")
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

#Title of website
st.title("Spotify Listener Skipping Behavior")

#Author names
st.markdown(
	"""
Created by Annie Fan, Brian Huang, Aishani Mohapatra, Victor Thai

""")

# #Sidebar 
# st.sidebar.button("Spotify Home",)

left, right = st.beta_columns(2)

with left:
	st.image("spotify_cover_image.jpg")
	st.caption("Spotify app displayed on smartphone.  " + "https://www.businessinsider.com/how-to-see-spotify-listening-history")

with right:
	st.header("Will You Skip The Next Song?")
	st.write("Predicting the skip behavior of listeners based on song genre...")

#EDA / Data Cleaning
st.header("EDA / Data Cleaning")

st.markdown("""
### Let's first look at our data...

With a total of 64 gb of user data, we sampled our data into training and test samples to apply our models... blah blah blah

	""")

log_data = pd.read_csv("Spotify_Data/data/training_set/log_mini.csv")
track_data = pd.read_csv("Spotify_Data/data/track_features/tf_mini.csv")

st.write(log_data)

st.write("Shown above is the sampled dataset...")

log_left, track_right = st.beta_columns(2)

with log_left:
	st.header("Log Data Columns")
	st.write(log_data.columns)

with track_right:
	st.header("Track Data Columns")
	st.write(track_data.columns)

st.write("To better understand the meaning of each column, follow this link. (test)")
column_def_url = 'https://www.ucsd.edu'
st.markdown(column_def_url, unsafe_allow_html=True)


