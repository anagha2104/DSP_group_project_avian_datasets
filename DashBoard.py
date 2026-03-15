import streamlit as st
import pandas as pd
import numpy as np

# 1. Load the AVONET dataset from the GitHub repository folder
@st.cache_data
def load_data():
    # Make sure this matches the exact file name your teammate uploaded
    df = pd.read_csv('data/raw/core/AVONET1_BirdLife.csv') 
    return df

df = load_data()

st.title("🔍 Bird Trait Lookup")

# 2. Create the searchable dropdown menu
# Note: Check if your teammate's CSV uses 'Species', 'Species1', or something else.
bird_list = df['Species1'].unique()

selected_bird = st.selectbox(
    "Type or select a bird species to view its traits:",
    options=bird_list
)

# 3. Filter the dataset based on the user's selection
bird_data = df[df['Species1'] == selected_bird]

# 4. Display the results
st.subheader(f"Data for: {selected_bird}")

# Option A: Show the raw table row
st.dataframe(bird_data)

# Option B: Show a few specific traits beautifully (Optional but looks great!)
st.write("### Key Measurements")
col1, col2, col3 = st.columns(3)
with col1:
    # .iloc[0] grabs the first matching row's specific column value
    st.metric(label="Body Mass (g)", value=bird_data['Mass'].iloc[0]) 
with col2:
    st.metric(label="Wing Length (mm)", value=bird_data['Wing.Length'].iloc[0])
with col3:
    st.metric(label="Beak Length (mm)", value=bird_data['Beak.Length_Culmen'].iloc[0])