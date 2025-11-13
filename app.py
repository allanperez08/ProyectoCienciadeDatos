import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Cargar el Modelo y las Columnas ---
try:
    model = joblib.load('modelo_prestamos.pkl')
    columnas_modelo = joblib.load('columnas_modelo.pkl')
    columnas_modelo = list(columnas_modelo)
except FileNotFoundError:
    st.error("Error: Archivos del modelo no encontrados ('modelo_prestamos.pkl' o 'columnas_modelo.pkl').")
    st.error("Asegúrate de ejecutar el notebook 'entrenamiento.ipynb' (Fase 2.5) primero.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar los archivos: {e}")
    st.stop()

# --- 2. Diccionarios de Traducción (Para la App) ---

# Para traducir 'Grade' (Letra a Número)
GRADE_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}

# Para 'Tipo de Vivienda' (Español a Inglés que el modelo espera)
VIVIENDA_MAP_INV = {
    'Alquiler (RENT)': 'RENT',
    'Hipoteca (MORTGAGE)': 'MORTGAGE',
    'Propia (OWN)': 'OWN'
}

# Para 'Estado de Verificación' (Español a Inglés)
VERIFICACION_MAP_INV = {
    'No Verificado': 'Not Verified',
    'Verificado (Fuente)': 'Source Verified',
    'Verificado': 'Verified'
}

# Para 'Propósito' (Español a Inglés)
PURPOSE_MAP_INV = {
    'Consolidación de Deuda': 'debt_consolidation',
    'Tarjeta de Crédito': 'credit_card',
    'Mejoras del Hogar': 'home_improvement',
    'Otro': 'other',
    'Compra Importante': 'major_purchase',
    'Médico': 'medical',
    'Pequeño Negocio': 'small_business',
    'Compra de Auto': 'car',
    'Mudanza': 'moving',
    'Vacaciones': 'vacation',
    'Casa': 'house',
    'Boda': 'wedding',
    'Energía Renovable': 'renewable_energy',
    'Educacional': 'educational'
}
# Creamos las listas de opciones en español para los selectores
opciones_vivienda_es = list(VIVIENDA_MAP_INV.keys())
opciones_verificacion_es = list(VERIFICACION_MAP_INV.keys())
opciones_purpose_es = list(PURPOSE_MAP_INV.keys())
opciones_grade = list(GRADE_MAP.keys())


# --- 3. Configuración de la Página ---
st.set_page_config(page_title="Simulador de Préstamos", page_icon="🏦")
st.title("Simulador de Aprobación de Préstamos 🏦")
st.write("""
Esta app usa un modelo de Machine Learning (LGBM) entrenado con **1.3 millones** de préstamos 
históricos para predecir el riesgo de incumplimiento.
""")

# --- Dashboard de Power BI Integrado ---
st.subheader("Dashboard de Préstamos (Power BI)")

powerbi_iframe = """
<iframe title="prestamos_dashboard" width="1000" height="600"
src="https://app.powerbi.com/view?r=eyJrIjoiOWEzMmQwYjItYWM0NC00OGJkLTk2MzgtZTk0NDE4Y2I3YTM1IiwidCI6IjBmNzg1NDlkLTNlZWMtNDNhZi1iNTZhLTZmN2IwNDJkNmM5YSIsImMiOjR9&pageName=c89659972260d55d077d"
frameborder="0" allowFullScreen="true"></iframe>
"""

st.components.v1.html(powerbi_iframe, height=650, scrolling=True)


# --- 4. Barra Lateral con Inputs del Usuario (¡AHORA TODO ES INPUT!) ---
st.sidebar.header("Datos del Solicitante")

# ¡CAMBIO! Todos son 'number_input' para teclear
loan_amnt = st.sidebar.number_input('Monto del Préstamo ($)', min_value=1000, max_value=40000, value=15000, step=500)
annual_inc = st.sidebar.number_input('Ingreso Anual ($)', min_value=10000, max_value=5000000, value=75000, step=1000)
int_rate = st.sidebar.number_input('Tasa de Interés (%)', min_value=5.0, max_value=31.0, value=12.5, step=0.1)
dti = st.sidebar.number_input('Ratio Deuda/Ingreso (DTI)', min_value=0.0, max_value=50.0, value=18.0, step=0.1)
open_acc = st.sidebar.number_input('Cuentas Abiertas (Crédito)', min_value=1, max_value=50, value=10, step=1)
pub_rec_bankruptcies = st.sidebar.number_input('Bancarrotas Registradas', min_value=0, max_value=3, value=0, step=1)

# Selectores (menús desplegables)
term = st.sidebar.selectbox('Plazo (Meses)', [36, 60])
grade = st.sidebar.selectbox('Calificación Crediticia (Grade)', options=opciones_grade) # ¡CAMBIO! de select_slider a selectbox
home_ownership_es = st.sidebar.selectbox('Tipo de Vivienda', options=opciones_vivienda_es)
verification_status_es = st.sidebar.selectbox('Estado de Verificación', options=opciones_verificacion_es)
purpose_es = st.sidebar.selectbox('Propósito del Préstamo', options=opciones_purpose_es)


# --- 5. Lógica de Predicción ---
if st.sidebar.button("Predecir Riesgo de Incumplimiento"):
    
    # 1. Crear un DataFrame de entrada con ceros
    data_entrada = pd.DataFrame(columns=columnas_modelo)
    data_entrada.loc[0] = np.zeros(len(columnas_modelo))
    
    # 2. Traducir las entradas en español a inglés (que el modelo espera)
    home_ownership_en = VIVIENDA_MAP_INV[home_ownership_es]
    verification_status_en = VERIFICACION_MAP_INV[verification_status_es]
    purpose_en = PURPOSE_MAP_INV[purpose_es]

    # 3. Asignar los valores del usuario a las columnas correctas
    data_entrada['loan_amnt'] = loan_amnt
    data_entrada['term'] = term
    data_entrada['int_rate'] = int_rate
    data_entrada['grade'] = GRADE_MAP[grade] # Traducimos 'C' a 2
    data_entrada['annual_inc'] = annual_inc
    data_entrada['dti'] = dti
    data_entrada['open_acc'] = open_acc
    data_entrada['pub_rec_bankruptcies'] = pub_rec_bankruptcies
    
    # 4. Asignar los valores categóricos (One-Hot Encoding)
    home_col = 'home_ownership_' + home_ownership_en
    if home_col in data_entrada.columns:
        data_entrada[home_col] = True
        
    verif_col = 'verification_status_' + verification_status_en
    if verif_col in data_entrada.columns:
        data_entrada[verif_col] = True
        
    purpose_col = 'purpose_' + purpose_en
    if purpose_col in data_entrada.columns:
        data_entrada[purpose_col] = True

    # 5. Mostrar los datos procesados (opcional)
    st.subheader("Datos de Entrada (Procesados para el Modelo):")
    st.dataframe(data_entrada[columnas_modelo].astype(str))

    # 6. Hacer la predicción
    try:
        prediccion = model.predict(data_entrada[columnas_modelo])[0]
        probabilidad = model.predict_proba(data_entrada[columnas_modelo])[0]
        
        riesgo = probabilidad[1] # Probabilidad de incumplimiento (Clase 1)
        
        # 7. Mostrar el resultado (en español)
        st.subheader("Resultado de la Evaluación:")
        if prediccion == 0:
            st.success(f"Resultado: PRÉSTAMO APROBADO (Bajo Riesgo)")
            st.write(f"Probabilidad de Incumplimiento: **{riesgo * 100:.2f}%**")
        else:
            st.error(f"Resultado: PRÉSTAMO RECHAZADO (Alto Riesgo)")
            st.write(f"Probabilidad de Incumplimiento: **{riesgo * 100:.2f}%**")
            
    except Exception as e:
        st.error(f"Error al predecir: {e}")
        st.error("Asegúrate de que las columnas del modelo coincidan con los datos de entrada.")

st.sidebar.info("Este modelo (LGBM) fue re-entrenado usando 'Sub-muestreo Aleatorio' para mejorar la detección de préstamos riesgosos.")
