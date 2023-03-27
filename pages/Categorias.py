import pickle
import shelve
import numpy as np
import pandas as pd
import streamlit as st
from random import randint
from sklearn.base import BaseEstimator, TransformerMixin

# Definiciones y datos necesarios
class CustomFeatureSelection(BaseEstimator, TransformerMixin):
    
    def __init__(self, selected_features):
        self.selected_features=selected_features
    
    def fit(self,X,y=None):
        return self
    
    def transform(self, X, y=None):
        return X[self.selected_features]

## Dataframe con los datos
with open('df_categorias.pkl', 'rb') as f_df:
    df = pickle.load(f_df)

with open('df_columns.pkl', 'rb') as f_columns:
    columns = pickle.load(f_columns)
    columns.remove('Categoria')
    columns.append('index')

## Modelos
opened_models = shelve.open('modelos_entrenados')
category_best_estimator = opened_models['modelo_categorias']
subcategory_best_estimator = opened_models['modelo_subcategorias']

st.title("Publicaciones de diferentes categorías")
st.subheader("Crea tu propia predicción")

def custom_selectbox(title, possible_values):
    return st.selectbox(title, possible_values, index=randint(0, len(possible_values)-1))

def custom_slider(title, minimum, maximum):
    return st.slider(title, int(minimum), int(maximum))

# listing_type_id
listing_type_id = custom_selectbox('Tipo de publicación', df.listing_type_id.unique().tolist())

# domain_id
domain_id = custom_selectbox('Dominio de la publicación', df.domain_id.unique().tolist())

# price
price = custom_slider("Precio de la publicación", df.price.min(), df.price.max())

# sold_quantity
sold_quantity = custom_slider("Cantidad de unidades vendidas", df.sold_quantity.min(), df.sold_quantity.max())

# available_quantity
available_quantity = custom_slider("Cantidad de unidades disponibles", df.available_quantity.min(), df.available_quantity.max())

# shipping__logistic_type
shipping__logistic_type = custom_selectbox('Tipo de logística', df.shipping__logistic_type.unique().tolist())

# shipping__mode
shipping__mode = custom_selectbox('Tipo de envío', df.shipping__mode.unique().tolist())

# shipping__store_pick_up
shipping__store_pick_up = st.checkbox('Store pick up', value=False)

# shipping__free_shipping
shipping__free_shipping = st.checkbox('Envio gratis', value=False)

# shipping__tags
shipping__tags = custom_selectbox('Tags de envío', df.shipping__tags.unique().tolist())

# installments__quantity
installments__quantity = custom_slider('Cantidad de installments', df.installments__quantity.min(), df.installments__quantity.max())

# installments__amount
installments__amount = custom_slider("Amount?? de installments", df.installments__amount.min(), df.installments__amount.max())

# installments__rate
installments__rate = custom_slider("Tasa de installments", df.installments__rate.min(), df.installments__rate.max())

# days_remaining
days_remaining = custom_slider("Días restantes", df.days_remaining.min(), df.days_remaining.max())

# years_active
years_active = custom_slider("Años activo", df.years_active.min(), df.years_active.max())


if 'result' not in st.session_state:
    st.session_state.result = None

def predict_with_data():
    x_pred = pd.DataFrame({
        "listing_type_id": listing_type_id,
        "domain_id": domain_id,
        "price": price,
        "sold_quantity": sold_quantity,
        "available_quantity": available_quantity,
        "shipping__logistic_type": shipping__logistic_type,
        "shipping__mode": shipping__mode,
        "shipping__store_pick_up": shipping__store_pick_up,
        "shipping__free_shipping": shipping__free_shipping,
        "shipping__tags": shipping__tags,
        "installments__quantity": installments__quantity,
        "installments__amount": installments__amount,
        "installments__rate": installments__rate,
        "days_remaining": days_remaining,
        "years_active": years_active,
        'variation_filters': np.nan},
        index=[0]
    )
    x_pred = x_pred.reset_index(drop=True)
    x_pred = pd.get_dummies(x_pred, columns=['domain_id', 'listing_type_id', 'shipping__logistic_type', 'shipping__mode', 'shipping__tags', 'variation_filters'])
    for col in columns:
        if col not in x_pred.columns:
            x_pred[col] = 0
    st.session_state.result = category_best_estimator.predict(x_pred)


if st.button(label="Predecir", on_click=predict_with_data):
    st.markdown(f"La publicación es de la categoría: **:blue{st.session_state.result}**")
