import shelve
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

class FeatureSelection(BaseEstimator, TransformerMixin):
    
    def __init__(self,selected_features):
        self.selected_features=selected_features
    
    def fit(self,X,y=None):
        return self
    
    def transform(self, X, y=None):
        return X[self.selected_features]

class CustomFeatureSelection(BaseEstimator, TransformerMixin):
    
    def __init__(self, selected_features):
        self.selected_features=selected_features
    
    def fit(self,X,y=None):
        return self
    
    def transform(self, X, y=None):
        return X[self.selected_features]
    

st.title(":blue[Presentación Trabajo Final: predicciones sobre publicaciones de Mercado Libre]")
st.write("Esta aplicación fue desarrollada en el marco del curso de Data Science de Digital House como trabajo final. Nuestro objetivo fue crear a partir de la API de Mercado Libre dos datasets diferentes, uno con informacion de publicaciones de productos de distintas categorías y otro con inmuebles publicados.")
st.write("**Entrenamos tres modelos que predicen:**")

st.markdown('- Las categorías de los productos')
st.markdown('- Las subcategorías de los productos')
st.markdown('- El precio de los inmuebles')

st.subheader("Data categorías y subcategorías")


with open('df_categorias.pkl', 'rb') as f_df:
    df = pickle.load(f_df)

st.write(df)

st.subheader("Data inmuebles")

import pandas as pd 
with open('df_inmuebles.pkl', 'rb') as i_df:
    df2 = pd.read_pickle(i_df)
    
st.write(df2)
