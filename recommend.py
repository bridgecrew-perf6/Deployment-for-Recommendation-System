# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 10:53:58 2022

@author: Admin
"""

import pandas as pd
import numpy as np

import streamlit as st
import pickle

import nltk
import sklearn

from sklearn.decomposition import TruncatedSVD

data = pd.read_csv('newdata.csv')
columns = ['user_id', 'song', 'listen_count']
new_df = data[['user_id', 'song', 'listen_count']]

ls_crosstab = new_df.pivot_table(values='listen_count', index='user_id', columns='song', fill_value=0)
X = ls_crosstab.T
SVD = TruncatedSVD(n_components=3, random_state=5)
resultant_matrix = SVD.fit_transform(X)
corr_matrix = np.corrcoef(resultant_matrix)

def get_recommendations(song_name):
    col_idx = ls_crosstab.columns.get_loc(song_name)
    corr_specific = corr_matrix[col_idx]
    sim_scores = list(enumerate(corr_specific))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    indices = [i[0] for i in sim_scores]
    return data['song'].iloc[indices]


st.title("Music Recommendation System")
from PIL import Image
img = Image.open("img.jpg")
 
# display image using streamlit
# width is used to set the width of an image
st.image(img, width=700)


##songs = pickle.load(open('song_list.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

song_list = new_df['song'].values
selected_song = st.selectbox("Type or select a song from the dropdown",song_list)

##sel = st.text_input(selected_song)
if st.button('Search'):
    st.write(selected_song)
    recommended_song_names = get_recommendations(selected_song)
    st.subheader("Recommended songs For You")
    for i in recommended_song_names:
        st.write(i)



