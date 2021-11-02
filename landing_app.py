import streamlit as st
import pandas as pd

#creates wider page setup
st.set_page_config(layout="wide")

spotify_image_left, spotify_image_right = st.beta_columns([1,8])

with spotify_image_left:
	spotify_logo = st.image("spotify_logo_image.png")

intro_left, intro_right = st.beta_columns([3,1])

with intro_left:
	st.markdown("#  Will You Skip The Next Song?")
	st.write("This project focues on analyzing the recommender systems among \
		Spotify's application, a popular music streaming service. To determine\
		 how Spotify is able to recommend songs to their listeners, we will \
		 look at specific musical traits among tracks that listeners \
		 have listened to, in order to predict whether they will play \
		 through or skip a particular song.")


bar_leftspacer, music_bar_left, music_bar, music_bar_right, bar_rightspacer = st.beta_columns([10,1.5,1.5,1.5,10])

with music_bar:
	play_button = st.image("spotify_play_button.png")
	if play_button:
		play_button = st.image("pause_button.png")

