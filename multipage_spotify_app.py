import streamlit as st
from multipage import save, MultiPage, start_app, clear_cache
import pandas as pd
from PIL import Image
import os

start_app() #Clears the cache when the app is started

app = MultiPage()
app.start_button = "Let's go!"
app.navbar_name = "Navigation"
app.next_page_button = "Next Page"
app.previous_page_button = "Previous Page"


def startpage():
	st.markdown("""# streamlit-multipage-framework
Framework for implementing multipage structure in streamlit apps.
It was inspired by upraneelnihar's project: https://github.com/upraneelnihar/streamlit-multiapps.
Developed by: Yan Almeida.
# Required Libraries
1. Streamlit (`pip install streamlit`);
2. Joblib (`pip install joblib`);
3. OS (`pip install os`).
# Code Elements
## Functions and Classes
1. function `initialize()` -> Runs when the program starts and sets the initial page as 0;
2. function `save(var_list, name, page_names)` -> Saves a list of variables, associates it with a name and defines which pages will receive these variables;
3. function `load(name)` -> Loads a var_list previously saved;
4. function `clear_cache(name=None)` -> Clears the variables in cache. Receives a list of variables to erase, but if none is given, clears all of the variables;
5. function `start_app()` -> Clears all the variables in the cache when the app is started (but not after this);
6. function `change_page(pag)` -> Sets the current page number as `pag`;
7. function `read_page()` -> Returns current page number;
8. class `app` -> Class to create pages (apps), defined by two attributes: name and func (app script defined as a function in the code);
9. class `MultiPage` -> Class to create the MultiPage structure, defined by the following attributes: apps (a list containing the pages (apps)), initial_page (used to set a starting page for the app, if needed), initial_page_set (used to determine whether a starting page is set or not), next_page_button and previous_page_button (in order to define the label of the buttons that switch between pages), navbar_name (to set the navigation bar header) and block_navbar (to keep your app without a navigation bar).
## MultiPage Public Attributes
1. `next_page_button` -> Defines the label of the "Next Page" button. Default: "Next Page";
2. `previous_page_button` -> Defines the label of the "Previous Page" button. Default: "Previous Page";
3. `start_button` -> Defines the label of the starting page button that starts the application (it's only used if the app has a starting page). Default: "Let's go!";
4. `navbar_name` -> Defines the Navigation Bar's name. Default: "Navigation".
## MultiPage Class Methods
1. `add_app(self, name, func)` -> Creates an app and adds it to the `apps` attribute;
2. `set_initial_page(self, func)` -> Sets a starting page to the program;
3. `disable_navbar(self)` -> Removes the navigation bar;
4. `run(self)` -> Creates a sidebar with buttons to switch between pages and runs the apps depending on the chosen page. It also keeps the variables defined in previous pages, if the app function correctly applies "save".
# How to use it
1. Download "multipage.py" and put it in the same folder as your app;
2. Import the class `MultiPage` and the functions `save` and `start_app` from multipage.py;
3. Create a `MultiPage` object;
4. Use the function `start_app` to clear the cache;
5. Set the buttons' labels (next_page_button and previous_page_button attributes) and the navigation bar name (navbar_name attribute);
6. Define the different pages (apps) as functions (use the `save` method in the end of each function if you need the app to remember the variables). If you do save variables, they are going to be passed as argument to the target functions;
7. Use the `add_app` method to include each one of the functions;
8. If you have a starting page for your program, include it by using the `set_initial_page` method;
9. Use the `run` method.""")

def landing(prev_vars): #Home Page

	spotify_image_left, spotify_image_right = st.columns([1,8])

	with spotify_image_left:
		spotify_logo = st.image("spotify.png")

	intro_left, intro_right = st.columns([3,1])

	with intro_left:
		st.markdown("#  Will You Skip The Next Song?")
		st.write("This project focues on analyzing the recommender systems among \
			Spotify's application, a popular music streaming service. To determine\
			 how Spotify is able to recommend songs to their listeners, we will \
			 look at specific musical traits among tracks that listeners \
			 have listened to, in order to predict whether they will play \
			 through or skip a particular song.")


	st.markdown('#')
	st.markdown('#')
	st.markdown('#')
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




def eda(prev_vars): #EDA / Data Cleaning
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


def discussion(prev_vars): #problems/problems
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



def about_us(prev_vars): #About Us Page
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




app.set_initial_page(startpage)
app.add_app("Home", landing)
app.add_app("EDA", eda)
app.add_app("Discussion", discussion)
app.add_app("About Us", about_us)

# app.add_app("Home", landing) #Adds first page (home) to the framework
# app.add_app("EDA", eda) #Adds second page (eda) to the framework
# app.add_app("Model", model) #Adds third page (model) to the framework
# app.add_app("Discussion", problems) #Adds fourth page (discussion) to the framework
# app.add_app("About Us", about_us) #Adds last page (about us) to the framework
app.run() #Runs the multipage app!
