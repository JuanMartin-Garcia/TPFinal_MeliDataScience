
import streamlit as st
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin



class FeatureSelection(BaseEstimator, TransformerMixin):
    
    def __init__(self,selected_features):
        self.selected_features=selected_features
    
    def fit(self,X,y=None):
        return self
    
    def transform(self, X, y=None):
        return X[self.selected_features]

#Loading up the Regression model we created
with open("./TheChosenOne.pkl", 'rb') as rfregr:
          modelo_regressor = pickle.load(rfregr)

variables_string_input =['seller__id', 'seller__real_estate_agency', 'seller__seller_reputation__transactions__canceled', 'seller__seller_reputation__transactions__completed',"seller__seller_reputation__metrics__sales__completed", 'seller__seller_reputation__transactions__ratings__negative', 'seller__seller_reputation__transactions__ratings__neutral', 'seller__seller_reputation__transactions__ratings__positive', 'seller__seller_reputation__transactions__total', 'seller__seller_reputation__metrics__claims__value', 'seller__seller_reputation__metrics__delayed_handling_time__rate', 'seller__seller_reputation__metrics__delayed_handling_time__value', 'seller__seller_reputation__metrics__cancellations__value', 'location__latitude', 'location__longitude', 'days_remaining', 'years_active', "domain_id_MLA-APARTMENTS_FOR_RENT",'domain_id_MLA-APARTMENTS_FOR_VACATION_RENTAL', 'domain_id_MLA-DEVELOPMENT_APARTMENTS_FOR_SALE', 'domain_id_MLA-DEVELOPMENT_HOUSES_FOR_SALE', 'domain_id_MLA-DEVELOPMENT_LANDS_FOR_SALE', 'domain_id_MLA-DOCTORS_PRACTICE_FOR_RENT', 'domain_id_MLA-FARMS_FOR_RENT', 'domain_id_MLA-FARMS_FOR_SALE', 'domain_id_MLA-FARM_HOUSES_FOR_SALE', 'domain_id_MLA-FARM_HOUSES_FOR_VACATION_RENTAL', 'domain_id_MLA-GARAGES_FOR_RENT', 'domain_id_MLA-GARAGES_FOR_SALE', 'domain_id_MLA-HORIZONTAL_PROPERTY_FOR_RENT', 'domain_id_MLA-HORIZONTAL_PROPERTY_FOR_SALE', 'domain_id_MLA-HORIZONTAL_PROPERTY_FOR_VACATION_RENTAL', 'domain_id_MLA-HOUSES_FOR_RENT', 'domain_id_MLA-HOUSES_FOR_VACATION_RENTAL', 'domain_id_MLA-INDIVIDUAL_APARTMENTS_FOR_SALE', 'domain_id_MLA-INDIVIDUAL_HOUSES_FOR_SALE', 'domain_id_MLA-INDIVIDUAL_LANDS_FOR_SALE', 'domain_id_MLA-INDIVIDUAL_OFFICES_FOR_SALE', 'domain_id_MLA-LANDS_FOR_RENT', 'domain_id_MLA-OFFICES_FOR_RENT', 'domain_id_MLA-OTHER_PROPERTIES_FOR_RENT', 'domain_id_MLA-OTHER_PROPERTIES_FOR_SALE', 'domain_id_MLA-RETAIL_SPACE_FOR_RENT', 'domain_id_MLA-RETAIL_SPACE_FOR_SALE', 'domain_id_MLA-TIMESHARE_PROPERTY_FOR_SALE', 'domain_id_MLA-TIMESHARE_PROPERTY_FOR_VACATION_RENTAL', 'domain_id_MLA-WAREHOUSES_FOR_RENT', 'domain_id_MLA-WAREHOUSES_FOR_SALE', 'currency_id_ARS', 'currency_id_USD']
# Define the prediction function
def predict(variables_input, variables_string_input):

    prediction = modelo_regressor.predict(pd.DataFrame([variables_input], columns=[variables_string_input], index=[0]))
    return prediction


st.title('Predictor Precio Inmueble MELI')
st.image("""https://http2.mlstatic.com/D_NQ_NP943608-MLA50100139306_052022-F.jpg""")
st.header('Indique las caracteristicas del inmueble:')
seller__id = st.number_input('Id Vendedor: (Please enter a 9 digit value)', min_value=1, max_value=999999999, value=123456789)
seller__real_estate_agency = st.selectbox('Es una inmobiliaria: (Bool)', [True, False])
seller__seller_reputation__transactions__canceled = st.number_input('Transacciones Canceladas: (Enter how many transactions where cancelled)', min_value=0, max_value=9999, value=1)
seller__seller_reputation__transactions__completed = st.number_input('Transacciones Completas: (Enter how many transactions where completed)', min_value=0, max_value=9999, value=1)
seller__seller_reputation__transactions__ratings__negative = st.number_input('Rating Transacciones Negativo: (Enter negative seller rating)', min_value=0.1, max_value=1.0, value=0.5)
seller__seller_reputation__transactions__ratings__neutral = st.number_input('Rating Transacciones Neutral: (Enter neutral seller rating)', min_value=0.1, max_value=1.0, value=0.5)
seller__seller_reputation__transactions__ratings__positive = st.number_input('Rating Transacciones Positivos: (Enter positive seller rating)', min_value=0.1, max_value=1.0, value=0.5)
seller__seller_reputation__transactions__total = st.number_input('Transacciones totales: (Enter how many transactions in total)', min_value=0, max_value=9999, value=1)
seller__seller_reputation__metrics__sales__completed = st.number_input('Ventas completas: (Enter how many sales were completed)', min_value=0, max_value=9999, value=1)
seller__seller_reputation__metrics__claims__value = st.number_input('Reclamos del Vendedor: (Enter how many claims the seller had)', min_value=0, max_value=9999, value=1)
seller__seller_reputation__metrics__delayed_handling_time__rate = st.number_input('Tiempo Promedio Respuesta: (Enter delayed handling time rate)', min_value=0.1, max_value=1.0, value=0.5)
seller__seller_reputation__metrics__delayed_handling_time__value = st.number_input('Tiempo de Respuesta: (Enter delayed handling time value)', min_value=0, max_value=9999, value=1)
seller__seller_reputation__metrics__cancellations__value = st.number_input('Valor de las Cancelaciones: (Enter how many cancellations the seller had)', min_value=0, max_value=9999, value=1)
location__latitude = st.number_input('location__latitude: (Enter the latitude of the property)', min_value=-9000000.0, max_value=9000000.0, value=0.0)
location__longitude = st.number_input('location__longitude: (Enter the longitude of the property)', min_value=-18000000.0, max_value=18000000.0, value=0.0)
days_remaining = st.number_input('Dias remanentes de la publicacion: (Enter how many days are left for the auction to end)', min_value=1, max_value=9999, value=1)
years_active = st.number_input('Años Activa: (Enter how many years the seller has been active)', min_value=1, max_value=9999, value=1)
domain_id_MLA_APARTMENTS_FOR_RENT = st.checkbox('APARTMENTS_FOR_RENT: (Check if this applies to your property)', value=0)
domain_id_MLA_APARTMENTS_FOR_VACATION_RENTAL = st.checkbox('APARTMENTS_FOR_VACATION_RENTAL: (Check if this applies to your property)', value=0)
domain_id_MLA_DEVELOPMENT_APARTMENTS_FOR_SALE = st.checkbox('DEVELOPMENT_APARTMENTS_FOR_SALE: (Check if this applies to your property)', value=0)
domain_id_MLA_DEVELOPMENT_HOUSES_FOR_SALE = st.checkbox('DEVELOPMENT_HOUSES_FOR_SALE: (Check if this applies to your property)', value=0)
domain_id_MLA_DEVELOPMENT_LANDS_FOR_SALE = st.checkbox('DEVELOPMENT_LANDS_FOR_SALE: (Check if this applies to your property)', value=0)
domain_id_MLA_DOCTORS_PRACTICE_FOR_RENT = st.checkbox('DOCTORS_PRACTICE_FOR_RENT: (Check if this applies to your property)', value=0)
domain_id_MLA_FARMS_FOR_RENT = st.checkbox('FARMS_FOR_RENT: (Check if this applies to your property)', value=0)
domain_id_MLA_FARMS_FOR_SALE = st.checkbox('FARMS_FOR_SALE: (Check if this applies to your property)', value=0)
domain_id_MLA_FARM_HOUSES_FOR_SALE = st.checkbox('FARM_HOUSES_FOR_SALE: (Check if this applies to your property)', value=0)
domain_id_MLA_FARM_HOUSES_FOR_VACATION_RENTAL = st.checkbox('FARM_HOUSES_FOR_VACATION_RENTAL: (Check if this applies to your property)', value=0)
domain_id_MLA_GARAGES_FOR_RENT = st.checkbox('GARAGES_FOR_RENT: (Check if this applies to your property)', value=0)
domain_id_MLA_GARAGES_FOR_SALE = st.checkbox('GARAGES_FOR_SALE: (Check if this applies to your property)', value=0)
domain_id_MLA_HORIZONTAL_PROPERTY_FOR_RENT = st.checkbox('HORIZONTAL_PROPERTY_FOR_RENT: (Check if this applies to your property)', value=0)
domain_id_MLA_HORIZONTAL_PROPERTY_FOR_SALE = st.checkbox('HORIZONTAL_PROPERTY_FOR_SALE: (Check if this applies to your property)', value=0)
domain_id_MLA_HORIZONTAL_PROPERTY_FOR_VACATION_RENTAL = st.checkbox('HORIZONTAL_PROPERTY_FOR_VACATION_RENTAL: (Check if this applies to your property)', value=0)
domain_id_MLA_HOUSES_FOR_RENT = st.checkbox('HOUSES_FOR_RENT: (Check if this applies to your property)', value=0)
domain_id_MLA_HOUSES_FOR_VACATION_RENTAL = st.checkbox('HOUSES_FOR_VACATION_RENTAL: (Check if this applies to your property)', value=0)
domain_id_MLA_INDIVIDUAL_APARTMENTS_FOR_SALE = st.checkbox('INDIVIDUAL_APARTMENTS_FOR_SALE: (Check if this applies to your property)', value=0)
domain_id_MLA_INDIVIDUAL_HOUSES_FOR_SALE = st.checkbox('INDIVIDUAL_HOUSES_FOR_SALE: (Check if this applies to your property)', value=0)
domain_id_MLA_INDIVIDUAL_LANDS_FOR_SALE = st.checkbox('INDIVIDUAL_LANDS_FOR_SALE: (Check if this applies to your property)', value=0)
domain_id_MLA_INDIVIDUAL_OFFICES_FOR_SALE = st.checkbox('INDIVIDUAL_OFFICES_FOR_SALE: (Check if this applies to your property)', value=0)
domain_id_MLA_LANDS_FOR_RENT = st.checkbox('LANDS_FOR_RENT: (Check if this applies to your property)', value=0)
domain_id_MLA_OFFICES_FOR_RENT = st.checkbox('OFFICES_FOR_RENT: (Check if this applies to your property)', value=0)
domain_id_MLA_OTHER_PROPERTIES_FOR_RENT = st.checkbox('OTHER_PROPERTIES_FOR_RENT: (Check if this applies to your property)', value=0)
domain_id_MLA_OTHER_PROPERTIES_FOR_SALE = st.checkbox('OTHER_PROPERTIES_FOR_SALE: (Check if this applies to your property)', value=0)
domain_id_MLA_RETAIL_SPACE_FOR_RENT = st.checkbox('RETAIL_SPACE_FOR_RENT: (Check if this applies to your property)', value=0)
domain_id_MLA_RETAIL_SPACE_FOR_SALE = st.checkbox('RETAIL_SPACE_FOR_SALE: (Check if this applies to your property)', value=0)
domain_id_MLA_TIMESHARE_PROPERTY_FOR_SALE = st.checkbox('TIMESHARE_PROPERTY_FOR_SALE: (Check if this applies to your property)', value=0)
domain_id_MLA_TIMESHARE_PROPERTY_FOR_VACATION_RENTAL = st.checkbox('TIMESHARE_PROPERTY_FOR_VACATION_RENTAL: (Check if this applies to your property)', value=0)
domain_id_MLA_WAREHOUSES_FOR_RENT = st.checkbox('WAREHOUSES_FOR_RENT: (Check if this applies to your property)', value=0)
domain_id_MLA_WAREHOUSES_FOR_SALE = st.checkbox('WAREHOUSES_FOR_SALE: (Check if this applies to your property)', value=0)
currency_id_ARS = st.checkbox('currency_id_ARS: (Check if this applies to your property)', value=0)
currency_id_USD = st.checkbox('currency_id_USD: (Check if this applies to your property)', value=0)

# Get a dictionary of all local variables in the current scope
#var_dict = locals()

variables_input=[seller__id, seller__real_estate_agency, seller__seller_reputation__transactions__canceled, seller__seller_reputation__transactions__completed,seller__seller_reputation__metrics__sales__completed, seller__seller_reputation__transactions__ratings__negative, seller__seller_reputation__transactions__ratings__neutral, seller__seller_reputation__transactions__ratings__positive, seller__seller_reputation__transactions__total, seller__seller_reputation__metrics__claims__value, seller__seller_reputation__metrics__delayed_handling_time__rate, seller__seller_reputation__metrics__delayed_handling_time__value, seller__seller_reputation__metrics__cancellations__value, location__latitude, location__longitude, days_remaining, years_active, domain_id_MLA_APARTMENTS_FOR_RENT, domain_id_MLA_APARTMENTS_FOR_VACATION_RENTAL, domain_id_MLA_DEVELOPMENT_APARTMENTS_FOR_SALE, domain_id_MLA_DEVELOPMENT_HOUSES_FOR_SALE, domain_id_MLA_DEVELOPMENT_LANDS_FOR_SALE, domain_id_MLA_DOCTORS_PRACTICE_FOR_RENT, domain_id_MLA_FARMS_FOR_RENT, domain_id_MLA_FARMS_FOR_SALE, domain_id_MLA_FARM_HOUSES_FOR_SALE, domain_id_MLA_FARM_HOUSES_FOR_VACATION_RENTAL, domain_id_MLA_GARAGES_FOR_RENT, domain_id_MLA_GARAGES_FOR_SALE, domain_id_MLA_HORIZONTAL_PROPERTY_FOR_RENT, domain_id_MLA_HORIZONTAL_PROPERTY_FOR_SALE, domain_id_MLA_HORIZONTAL_PROPERTY_FOR_VACATION_RENTAL, domain_id_MLA_HOUSES_FOR_RENT, domain_id_MLA_HOUSES_FOR_VACATION_RENTAL, domain_id_MLA_INDIVIDUAL_APARTMENTS_FOR_SALE, domain_id_MLA_INDIVIDUAL_HOUSES_FOR_SALE, domain_id_MLA_INDIVIDUAL_LANDS_FOR_SALE, domain_id_MLA_INDIVIDUAL_OFFICES_FOR_SALE, domain_id_MLA_LANDS_FOR_RENT, domain_id_MLA_OFFICES_FOR_RENT, domain_id_MLA_OTHER_PROPERTIES_FOR_RENT, domain_id_MLA_OTHER_PROPERTIES_FOR_SALE, domain_id_MLA_RETAIL_SPACE_FOR_RENT, domain_id_MLA_RETAIL_SPACE_FOR_SALE, domain_id_MLA_TIMESHARE_PROPERTY_FOR_SALE, domain_id_MLA_TIMESHARE_PROPERTY_FOR_VACATION_RENTAL, domain_id_MLA_WAREHOUSES_FOR_RENT, domain_id_MLA_WAREHOUSES_FOR_SALE, currency_id_ARS,currency_id_USD]


# Extract the variable names and save them to a list
#var_names = [var for var in var_dict.keys() if not var.startswith('_')]
#var_names_new = var_names[12:]

# Print the list of variable names
#print(var_names_new)

if st.button('Predict Price'):
    price = predict(variables_input,variables_string_input)
    st.success(f'El precio para el inmueble es: ${price[0]:.2f}') 

