import pandas as pd
import streamlit as st
import os


# creates a wider page on streamlit
st.set_page_config(layout="wide")

spotify_image_left, spotify_image_right = st.columns([1,8])

with spotify_image_left:
	spotify_logo = st.image("spotify.png")

#Title of Page
st.markdown("# Discussion of Methods")


#problems with sklearn
st.markdown("""## __Experiences with scikit-learn__	""")

st.image("spotify_streamlit_photos/sklearn.png")

st.markdown("""
Using sklearn to generate a model that would predict skipping behavior allowed for
data from different users to become a general prediction. In the real world, this
is not accurate as every listener has their own unique taste and skipping behavior.
In other words, we simply cannot assume that everyone shares the same behavior through
sampling from the whole dataset.

Rather, it must be recognized that each listener has their unique listening styles.
Therefore, it is essential that we sample the listening behavior of each specific listener
in order to determine traits that shape the skipping behavior of listeners.
	""")

#problems with behavior class
st.markdown("""## __Problems with the Behavior Class__""")

st.image("spotify_streamlit_photos/clustering.png")

st.markdown("""
The behavior class is the prediction model we created from scratch; we had much more
freedom to create our own models based on the specifications of what we wanted to do.
The main aspect we wanted to change from our sklearn model, was the fact that we
weren't able to group users' listening data together while predicting if a user
would skip a song. The prediction was generalized to all users in the dataset.

However, we still came across some issues with our behavior class, especially since
we had no starting framework and there were too many elements to account for that
were much more easily done using the sklearn model, which is eventually what
we used for our final model.

	""")

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
