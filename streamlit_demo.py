#Make sure to have streamlit downloaded first!

# https://streamlit.io/#install

#After downloading, open terminal and type "streamlit run streamlit_demo.py"

import pandas as pd
import streamlit as st
import yfinance as yf
from PIL import Image

# Background image of website
st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://pixy.org/src/6/61745.png")
    }
   .sidebar .sidebar-content {
        background: url("insert picture url here")
    }
    </style>
    """,
    unsafe_allow_html=True
)


#Title of website
st.title(" Simple Stock Price App ")
st.markdown("""

### Adapted from Data Professor on Youtube

Shown are the stock closing price and volume of stock of choice!

""")

#embedding images
st.image("wallstreet.jpg")

#input section for stock to look at
stock = st.text_input(" Enter stock code of choice: ")

#define the ticker symbol
tickerSymbol = stock

st.write("You have chosen " + stock + "!")

#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

#get the historical prices for this ticker
tickerDf = tickerData.history(period='ytd')
# Open	High	Low	Close	Volume	Dividends	Stock Splits
st.markdown(" ## Year To Date Trend Close Prices of " + stock)
st.line_chart(tickerDf.Close)


#daily close price
st.markdown(" ## Daily Close Prices")
data = yf.download(tickers = stock, period = '5d', interval = '1m')
data_close = data.get('Close')
st.line_chart(data_close)

#size of volume
st.markdown(" ## Size of Volume of Year")
st.line_chart(tickerDf.Volume)


