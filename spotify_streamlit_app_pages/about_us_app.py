import pandas as pd
import streamlit as st
import os


# creates a wider page on streamlit
st.set_page_config(layout="wide")

spotify_image_left, spotify_image_right = st.columns([1,8])

with spotify_image_left:
	spotify_logo = st.image("spotify_streamlit_photos/spotify.png")

#Title of Page
st.markdown("""## __About Our Team__

### Hello! We are undergraduate data science students at the University of California - San Diego
	""")


images = ["spotify_streamlit_photos/brian.jpg", "spotify_streamlit_photos/annie.jpg", "spotify_streamlit_photos/victor.jpg",
"spotify_streamlit_photos/aishani.jpg"]
#st.image(images, width=300, caption=["Brian H", "Annie Fan", "Victor Thai", "Aishani Mohapatra" ])
brian, annie, victor, aishani = st.columns(4)
with brian:
	brian = st.image(images[0])
	st.markdown("#### Brian Huang")
	st.write("My name is Brian and I was born June 28 in Singapore. I am currently a Data science major and psychology minor at Warren College. \
	I aspire to apply my data analysis skills to biology or medicine to speed up research processes. I have two sisters, Katherine and Angelina, and I enjoy playing tennis and cycling.")
with annie:
	annie = st.image(images[1])
	st.markdown('#### Annie Fan')
	st.write("My name is Annie Fan and I am a third year student majoring in data science and cognitive science. \
		I grew up in China and I attended high school in Seattle, Washington. I am into data analysis and machine learning. \
		Currently, I am learning data management with SQL and building recommender systems with data mining.")
with victor:
	victor = st.image(images[2])
	st.markdown('#### Victor Thai')
	st.write("My name is Victor and I am a second year studying Data Science at UCSD. I grew up in Oakland, CA and have just recently moved to San Diego to pursue my studies. \
		In my free time, I enjoy staying active and being in nature by going on hikes. I plan on using data science to create improvements to heatlthcare technologies and procedures.")
with aishani:
	aishani = st.image(images[3])
	st.markdown('#### Aishani Mohapatra')
	st.write("My name is Aishani and I am a second year at Muir College studying data science from San Ramon, CA. I am interested in exploring topics the intersection between language and machine learning, \
		 particularly with Natural Language Processing. In my free time, she enjoys singing and hiking with friends.")
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
st.progress(90)
