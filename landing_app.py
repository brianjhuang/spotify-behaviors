import streamlit as st
import pandas as pd

#creates wider page setup
st.set_page_config(layout="wide")

spotify_image_left, spotify_image_mid, spotify_image_right = st.columns([8,5,8])

with spotify_image_mid:
	spotify_logo = st.image("spotify.png")

intro_left, intro_mid, intro_right = st.columns([1,3,1])

with intro_mid:
	st.markdown("#  Will You Skip The Next Song?")
	st.markdown("""
		This project focuses on analyzing the recommender systems among \
		Spotify's application, a popular music streaming service. To determine\
		 how Spotify is able to recommend songs to their listeners and create \
		 personal playlists geared towards their listeners, we will \
		 look at specific musical traits among tracks that listeners \
		 have listened to, in order to predict whether they will play \
		 through or skip a particular song. """)


st.markdown('#')

bar_leftspacer, music_bar_left, music_bar, music_bar_right, bar_rightspacer = st.columns([10,1.5,1.5,1.5,10])

with music_bar:
	play_button = st.image("spotify_streamlit_photos/spotify_play_button.png")
	# if play_button:
	# 	play_button = st.image("pause_button.png")
with music_bar_right:
	st.image("spotify_streamlit_photos/skip_button_spotify.png", use_column_width = True)
st.progress(2)