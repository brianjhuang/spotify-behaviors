import pandas as pd
import streamlit as st
import os


# creates a wider page on streamlit
st.set_page_config(layout="wide")

spotify_image_left, spotify_image_right = st.columns([1,8])

with spotify_image_left:
	spotify_logo = st.image("spotify.png")

#Title of Page
st.markdown("""## __About Our Team__

### Hello! We are undergraduate data science students at the University of California - San Diego
	""")


images = ["spotify_streamlit_photos/brian.jpg", "spotify_streamlit_photos/annie.jpg", "spotify_streamlit_photos/victor.jpg",
"spotify_streamlit_photos/aishani.jpg"]
#st.image(images, width=300, caption=["Brian H", "Annie Fan", "Victor Thai", "Aishani Mohapatra" ])
brian, annie, victor, aishani = st.columns(4)
with brian:
	brian = st.image(images[0], caption = "Brian Huang")
with annie:
	annie = st.image(images[1], caption = "Annie Fan")
with victor:
	victor = st.image(images[2], caption = "Victor Thai")
with aishani:
	aishani = st.image(images[3], caption = "Aishani Mohapatra")
#st.image(images, use_column_width = False, caption=["Brian", "Annie", "Victor", "Aishani" ])

st.markdown('#')
st.markdown('#')

bar_leftspacer, music_bar_left, music_bar, music_bar_right, bar_rightspacer = st.columns([10,1.5,1.5,1.5,10])

with music_bar:
	play_button = st.image("spotify_streamlit_photos/spotify_play_button.png")
	# if play_button:
	# 	play_button = st.image("pause_button.png")
with music_bar_right:
	st.image("spotify_streamlit_photos/skip_button_spotify.png", use_column_width = True)
st.progress(2)
