import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.title("Avian Dashboard")
st.sidebar.header("Controls")
x_axis = st.sidebar.selectbox("X-Axis", ['BodyMass', 'WingLength'])
missing_strategy = st.sidebar.radio("Missing Data Strategy", ['Drop', 'Impute'])
run_pca = st.sidebar.button("Run PCA")

def load_data():
    # df = pd.read_csv('team_data.csv') 
    df = pd.DataFrame({})
    return df

fig = px.scatter(df_processed, x=x_axis, y="BodyMass")
st.plotly_chart(fig)

if run_pca:
    st.write("Running your team's code...")



