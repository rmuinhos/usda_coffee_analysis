import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Configuração da página
st.set_page_config(page_title="USDA Coffee Analisis", layout="wide")

# Mapeamento de Attribute ID para Descrição
attribute_descriptions = {
    29: "Arabica Production",
    90: "Bean Exports",
    58: "Bean Imports",
    20: "Beginning Stocks",
    125: "Domestic Consumption",
    176: "Ending Stocks",
    88: "Exports",
    57: "Imports",
    56: "Other Production",
    28: "Production",
    107: "Roast & Ground Exports",
    75: "Roast & Ground Imports",
    53: "Robusta Production",
    141: "Rst,Ground Dom. Consum",
    154: "Soluble Dom. Cons.",
    114: "Soluble Exports",
    82: "Soluble Imports",
    178: "Total Distribution",
    86: "Total Supply"
}

# Mapeamento de Códigos de País para Nome
countries = {
    "AL": "Albania",
    "AG": "Algeria",
    "AO": "Angola",
    "AR": "Argentina",
    "AM": "Armenia",
    "AS": "Australia",
    "DM": "Benin",
    "BL": "Bolivia",
    "BK": "Bosnia and Herzegovina",
    "BR": "Brazil",
    "BY": "Burundi",
    "CM": "Cameroon",
    "CA": "Canada",
    "CT": "Central African Republic",
    "CI": "Chile",
    "CH": "China",
    "CO": "Colombia",
    "CF": "Congo (Brazzaville)",
    "CG": "Congo (Kinshasa)",
    "CS": "Costa Rica",
    "IV": "Cote d'Ivoire",
    "HR": "Croatia",
    "CU": "Cuba",
    "DR": "Dominican Republic",
    "EC": "Ecuador",
    "EG": "Egypt",
    "ES": "El Salvador",
    "EK": "Equatorial Guinea",
    "ET": "Ethiopia",
    "E4": "European Union",
    "GB": "Gabon",
    "GG": "Georgia",
    "GH": "Ghana",
    "GT": "Guatemala",
    "GU": "Guinea",
    "GY": "Guyana",
    "HA": "Haiti",
    "HO": "Honduras",
    "IN": "India",
    "ID": "Indonesia",
    "IR": "Iran",
    "JM": "Jamaica",
    "JA": "Japan",
    "JO": "Jordan",
    "KZ": "Kazakhstan",
    "KE": "Kenya",
    "KS": "Korea, South",
    "KV": "Kosovo",
    "LA": "Laos",
    "LI": "Liberia",
    "MA": "Madagascar",
    "MI": "Malawi",
    "MY": "Malaysia",
    "MX": "Mexico",
    "MJ": "Montenegro",
    "MO": "Morocco",
    "NC": "New Caledonia",
    "NZ": "New Zealand",
    "NU": "Nicaragua",
    "NI": "Nigeria",
    "MK": "North Macedonia",
    "NO": "Norway",
    "PN": "Panama",
    "PP": "Papua New Guinea",
    "PA": "Paraguay",
    "PE": "Peru",
    "RP": "Philippines",
    "RS": "Russia",
    "RW": "Rwanda",
    "SA": "Saudi Arabia",
    "SG": "Senegal",
    "RB": "Serbia",
    "SL": "Sierra Leone",
    "SN": "Singapore",
    "SF": "South Africa",
    "CE": "Sri Lanka",
    "SZ": "Switzerland",
    "TW": "Taiwan",
    "TZ": "Tanzania",
    "TH": "Thailand",
    "TO": "Togo",
    "TD": "Trinidad and Tobago",
    "TU": "Turkey",
    "UG": "Uganda",
    "UP": "Ukraine",
    "UK": "United Kingdom",
    "US": "United States",
    "UY": "Uruguay",
    "VE": "Venezuela",
    "VM": "Vietnam",
    "YM": "Yemen",
    "YE": "Yemen (Sanaa)",
    "ZA": "Zambia",
    "RH": "Zimbabwe"
}

# Função para buscar dados a partir do ano e país
@st.cache_data
def fetch_data(year, country_code):
    url = f'https://api.fas.usda.gov/api/psd/commodity/0711100/country/{country_code}/year/{year}'
    headers = {
        'accept': 'application/json', 
        'X-Api-Key': 'eE1ladHVzY2StsA5efcXmaHKPfAa6XAjUetwVvHX'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Erro na conexão: {str(e)}")
        return None

# Sidebar
with st.sidebar:
    st.header('⚙️ Filters')
    # Filtro de Ano
    year_choice = st.selectbox('Year', options=['All'] + list(range(2010, 2025)), index=14)
    
    # Filtro de País: inclui a opção "Todos"
    country_choice = st.selectbox(
        'Country', 
        options=["all"] + list(countries.keys()), 
        format_func=lambda x: "All" if x == "all" else countries[x]
    )
    
    # Carregar atributos dinamicamente
    default_data = fetch_data(2024, country_choice) or []
    attributes = sorted({item['attributeId'] for item in default_data})
    attribute_choice = st.selectbox('Attribute', options=attributes, 
                                    format_func=lambda x: attribute_descriptions.get(x, f"Attribute {x}"))

# Conteúdo principal
st.title('📈 USDA Coffee - Coffee Analysis')
st.text("Ronaldo Muinhos - rmuinhos@gmail.com")

if year_choice == 'All':
    years = st.slider("Select Range", 2010, 2024, (2014, 2024))
    
    all_data = []
    progress_bar = st.progress(0)
    for i, year in enumerate(range(years[0], years[1]+1)):
        data = fetch_data(year, country_choice)
        if data:
            filtered = [item for item in data if item['attributeId'] == attribute_choice]
            all_data.extend(filtered)
        progress_bar.progress((i + 1)/(years[1] - years[0] + 1))
    
    if all_data:
        df = pd.DataFrame(all_data)
        df_grouped = df.groupby('marketYear', as_index=False).agg({'value': 'sum'})
        
        if len(df_grouped) > 1:
            # Modelagem de regressão
            X = df_grouped['marketYear'].values.reshape(-1, 1)
            y = df_grouped['value'].values
            
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Métricas
            r2 = r2_score(y, y_pred)
            equation = f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}"
            
            # Gráfico combinado
            title_description = attribute_descriptions.get(attribute_choice, f"Attribute {attribute_choice}")
            country_name = "All" if country_choice == "all" else countries[country_choice]
            fig = px.bar(df_grouped, x='marketYear', y='value', 
                         title=f'Trend - {title_description} ({country_name})',
                         labels={'value': 'Value', 'marketYear': 'Year'})
            
            fig.add_scatter(x=df_grouped['marketYear'], y=y_pred, 
                            mode='lines+markers', name='Trend',
                            line=dict(color='red', width=3))
            
            # Exibição
            col1, col2, col3 = st.columns(3)
            col1.metric("Slope", f"{model.coef_[0]:.2f}")
            col2.metric("Intercepto", f"{model.intercept_:.2f}")
            col3.metric("R² Score", f"{r2:.2%}")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Previsão
            st.subheader("Prediction for the Coming Years")
            future_years = st.slider("Future Years", 1, 5, 3)
            
            future_X = np.array(range(years[1]+1, years[1]+1+future_years)).reshape(-1, 1)
            future_pred = model.predict(future_X)
            
            pred_df = pd.DataFrame({
                'Year': future_X.flatten(),
                'Trend': future_pred
            })
            
            st.line_chart(pred_df.set_index('Year'))
        else:
            st.warning("Dados insuficientes para análise de tendência")
        
        with st.expander("View Detailed Data"):
            st.dataframe(df_grouped)
    else:
        st.warning("Nenhum dado encontrado para o período selecionado")

else:
    data = fetch_data(year_choice, country_choice)
    if data:
        filtered = [item for item in data if item['attributeId'] == attribute_choice]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Value", sum(item['value'] for item in filtered))
        with col2:
            st.metric("Records", len(filtered))
        
        st.dataframe(pd.DataFrame(filtered))

# Informações adicionais
st.divider()
st.markdown("""
**Model Interpretation:**
- **Slope Coefficient:** Expected annual growth rate
- **Intercept:** Theoretical baseline value for year zero
- **R² Score:** Goodness of fit (0-100%)
- **Unit:** (1000 60 KG BAGS)         
""")

st.divider()
st.markdown("""
Production, Supply & Distribution (PSD) Data
PSD Online is the public repository for USDA’s Official Production, Supply and Distribution forecast data, reports and circulars for key agricultural commodities.

FAS's PSD Online data are reviewed and updated monthly by an interagency committee chaired by USDA's World Agricultural Outlook Board (WAOB),and consisting of: the Foreign Agricultural Service (FAS), the Economic Research Service (ERS),the Farm Service Agency (FSA), and the Agricultural Marketing Service (AMS).

The international portion of the data is updated with input from agricultural attachés stationed at U.S. embassies around the world, FAS commodity analysts, and country and commodity analysts with ERS. The U.S. domestic component is updated with input from analysts in FAS, ERS, the National Agricultural Statistical Service, and FSA.
""")