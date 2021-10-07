import pandas as pd
import streamlit as st
import os


# creates a wider page on streamlit
st.set_page_config(layout="wide")

#Title of Page
st.markdown("""## __About Our Team__

### Hello! We are undergraduate students at the University of California - San Diego
	""")


images = ["images.png", "images.png", "images.png", "images.png"]
st.image(images, use_column_width=True, caption=["Brian", "Annie", "Victor", "Aishani" ])
