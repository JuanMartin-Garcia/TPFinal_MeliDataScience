import shelve
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
import pickle


class CustomFeatureSelection(BaseEstimator, TransformerMixin):
    
    def __init__(self, selected_features):
        self.selected_features=selected_features
    
    def fit(self,X,y=None):
        return self
    
    def transform(self, X, y=None):
        return X[self.selected_features]
    

st.title("Presentación del Trabajo Final: predicciones sobre publicaciones de Mercado Libre")
st.write("Data Categorías y subcategorías")

with open('df_categorias.pkl', 'rb') as f_df:
    df = pickle.load(f_df)

st.write(df)

