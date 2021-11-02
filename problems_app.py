import pandas as pd
import streamlit as st
import os


# creates a wider page on streamlit
st.set_page_config(layout="wide")

#Title of Page
st.markdown("# Discussion")


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

#problems with behavior class
st.markdown("""## __Problems with the Behavior Class__

The behavior class is the prediction model we created from scratch; we had much more
freedom to create our own models based on the specifications of what we wanted to do.
However, we still came across some issues, especially since we had no starting framework
and had to start everything from scratch.

	""")

st.button("images.png")
