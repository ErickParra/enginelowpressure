import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyodbc
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Acceder a los secrets almacenados en Streamlit Cloud
server = st.secrets["server"]
database = st.secrets["database"]
username = st.secrets["username"]
password = st.secrets["password"]

@st.cache
def load_data(query, conn_str):
    try:
        conn = pyodbc.connect(conn_str)
        data = pd.read_sql(query, conn)
        conn.close()
        return data
    except pyodbc.Error as e:
        st.error(f"Error al conectar a la base de datos: {e}")
        return pd.DataFrame()  # Retorna un DataFrame vacío en caso de error

# Configuración de la conexión a la base de datos
conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

# Ejecución de la consulta SQL
query = """
SELECT
       [EquipmentName]
      ,[ReadTime]
      ,[EquipmentModel]
      ,[ParameterName]
      ,[ParameterFloatValue]
  FROM [OemDataProvider].[OemParameterExternalView]
  WHERE ([EquipmentModel] = '797F')
              AND ParameterFloatValue IS NOT NULL
        AND ReadTime > (DATEADD (hour, -120, GETDATE()))
        AND (ParameterName =  'Engine Oil Pressure (Absolute)' OR
            ParameterName =  'Engine Oil Pressure' OR
            ParameterName =  'Engine Oil Pressure Front' OR
            ParameterName =  'Engine Oil Pressure Rear' OR
            ParameterName =  'Engine Speed' OR
            ParameterName =  'Engine Oil Pressure (High Resolution/Extended Range)')
"""

# Ejecutar la consulta y obtener los datos
with st.spinner('Ejecutando consulta...'):
    data = load_data(query, conn_str)
st.success('Consulta completada!')

# Verificar si los datos se han obtenido correctamente
if data.empty:
    st.error("No se pudo obtener datos de la base de datos. Revisa la conexión y los logs para más detalles.")
else:
    data = data.sort_values(by='ReadTime', ascending=False)
    st.write("### Datos obtenidos desde Minecare 3")
    st.dataframe(data)

    # Filtrar los datos para 797F y Engine Oil Pressure
    data_797F = data.loc[(data['EquipmentModel'] == '797F') & (data['ParameterName'] == 'Engine Oil Pressure')]

    # Graficar datos
    st.write("### Gráfico de Engine Oil Pressure 797F")
    fig, ax = plt.subplots(figsize=(16, 8))

    if not data_797F.empty:
        data_797F = data_797F[(data_797F['ParameterFloatValue'] >= 150) & (data_797F['ParameterFloatValue'] <= 1000)]
        group_797F = data_797F.groupby('EquipmentName')

        for name, group in group_797F:
            group.plot(x='ReadTime', y='ParameterFloatValue', label=name, ax=ax, linewidth=0.5)
    else:
        st.write("No hay datos disponibles para el modelo 797F y Engine Oil Pressure en el rango especificado.")

    plt.legend()
    st.pyplot(fig)

    # Resto del código para gráficos y análisis de datos...

    # Verificar si los datos se han obtenido correctamente
    st.write("### Datos obtenidos desde Minecare 3")
    st.dataframe(data)

    # Filtrar los datos para 797F y Engine Oil Pressure
    data_797F = data.loc[(data['EquipmentModel'] == '797F') & (data['ParameterName'] == 'Engine Oil Pressure')]

    # Graficar datos
    st.write("### Gráfico de Engine Oil Pressure 797F")
    fig, ax = plt.subplots(figsize=(16, 8))

    if not data_797F.empty:
        data_797F = data_797F[(data_797F['ParameterFloatValue'] >= 150) & (data_797F['ParameterFloatValue'] <= 1000)]
        group_797F = data_797F.groupby('EquipmentName')

        for name, group in group_797F:
            group.plot(x='ReadTime', y='ParameterFloatValue', label=name, ax=ax, linewidth=0.5)
    else:
        st.write("No hay datos disponibles para el modelo 797F y Engine Oil Pressure en el rango especificado.")

    plt.legend()
    st.pyplot(fig)

    # Graficar relación entre Engine Speed y Engine Oil Pressure
    st.write("### Relación entre Engine Speed (RPM) y Engine Oil Pressure (PSI)")
    PRESSURE_LVL3_PSI = [0, 100, 120, 225, 240, 255, 270, 285, 300, 309, 319, 329, 337.5, 346, 356, 366, 376]
    PRESSURE_LVL1_PSI = [100, 125, 155, 250, 270, 290, 310, 330, 350, 356, 361, 369, 375, 381, 387, 395, 400]
    RPM = [560, 580, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]

    X_pressure3 = np.linspace(600, 2000, len(PRESSURE_LVL3_PSI))
    X_pressure1 = np.linspace(600, 2000, len(PRESSURE_LVL1_PSI))

    data_speed = data.loc[(data['EquipmentModel'] == '797F') & (data['ParameterName'] == 'Engine Speed') & (data['ParameterFloatValue'] >= 200) & (data['ParameterFloatValue'] <= 5000)]
    data_pressure = data.loc[(data['EquipmentModel'] == '797F') & (data['ParameterName'] == 'Engine Oil Pressure') & (data['ParameterFloatValue'] >= 200) & (data['ParameterFloatValue'] <= 5000)]

    merged_data = pd.merge(data_speed, data_pressure, on=['ReadTime', 'EquipmentName'])

    fig, ax = plt.subplots(figsize=(16, 8))

    if not merged_data.empty:
        for name, group in merged_data.groupby('EquipmentName'):
            ax.scatter(group['ParameterFloatValue_x'], group['ParameterFloatValue_y'], label=name, alpha=0.5, s=0.75)
            ax.plot(X_pressure3, PRESSURE_LVL3_PSI, '--', color='orange', linewidth=0.5)
            ax.plot(X_pressure1, PRESSURE_LVL1_PSI, '--', color='gray', linewidth=0.5)
    else:
        st.write("No hay datos disponibles para la relación entre Engine Speed y Engine Oil Pressure en el rango especificado.")

    ax.set_xlabel('Engine Speed')
    ax.set_ylabel('Engine Oil Pressure')
    ax.legend()
    st.pyplot(fig)

    # Modelos de regresión y ajuste de curvas
    st.write("### Presión vs RPM ajuste de curva y limites L1/L3")

    grouped_data = merged_data.groupby('EquipmentName')
    models = {}

    for name, group in grouped_data:
        X = group['ParameterFloatValue_x'].values.reshape(-1, 1)
        y = group['ParameterFloatValue_y'].values
        
        poly_reg = PolynomialFeatures(degree=4)
        X_poly = poly_reg.fit_transform(X)
        
        lin_reg = LinearRegression()
        lin_reg.fit(X_poly, y)
        
        models[name] = (lin_reg, poly_reg)

    if not merged_data.empty:
        nrows = int(np.ceil(np.sqrt(len(grouped_data))))
        ncols = int(np.ceil(len(grouped_data) / nrows))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 8))

        for (name, group), ax in zip(grouped_data, axes.flat):
            X = group['ParameterFloatValue_x'].values.reshape(-1, 1)
            y = group['ParameterFloatValue_y'].values

            if name in models:
                lin_reg, poly_reg = models[name]

                X_grid = np.arange(X.min(), X.max(), 0.1).reshape(-1, 1)
                y_pred = lin_reg.predict(poly_reg.transform(X))
                r2 = r2_score(y, y_pred)

                formula = f"y = {lin_reg.intercept_:.2f}"
                for i, coef in enumerate(lin_reg.coef_[1:], start=1):
                    formula += f" + {coef:.2f}x^{i}"

                ax.scatter(X, y, color='red', alpha=0.5, s=1)
                ax.plot(X_grid, lin_reg.predict(poly_reg.transform(X_grid)), color='blue')
                ax.set_ylim(0, 750)

                ax.plot(X_pressure3, PRESSURE_LVL3_PSI, '--', color='orange', linewidth=0.5, label='Pressure Level 3', markersize=2) 
                ax.plot(X_pressure1, PRESSURE_LVL1_PSI, '--', color='gray', linewidth=0.5,   label='Pressure Level 1', markersize=2) 

                ax.set_title(f"{name} (R2={r2:.2f})")
                ax.set_xlabel('Engine Speed')
                ax.set_ylabel('Engine Oil Pressure')
                ax.text(0.05, 0.05, formula, transform=ax.transAxes, fontsize=5,
                        verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.8))
                ax.legend(fontsize=2)
            else:
                ax.set_title(f"{name} - Modelo no disponible")
                ax.set_xlabel('Engine Speed')
                ax.set_ylabel('Engine Oil Pressure')

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("No hay datos disponibles para los modelos de regresión y ajuste de curvas.")

    # Añadir las métricas calculadas al final
    results = []

    for name, group in grouped_data:
        X = group['ParameterFloatValue_x'].values.reshape(-1, 1)
        y = group['ParameterFloatValue_y'].values
        
        if name in models:
            lin_reg, poly_reg = models[name]
            y_pred = lin_reg.predict(poly_reg.transform(X))
            
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            
            results.append({
                "EquipmentName": name,
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae
            })

    # Convertir los resultados en un DataFrame
    results_df = pd.DataFrame(results)

    # Ordenar por RMSE de mayor a menor
    results_df = results_df.sort_values(by="RMSE", ascending=False)

    # Mostrar la tabla de resultados
    st.write("### Ranking métricas de Camiones")
    st.dataframe(results_df)

    # Filtrar datos solo donde Engine Speed es mayor a 750 RPM
    filtered_data = merged_data[merged_data['ParameterFloatValue_x'] >= 800]

    # Calcular métricas y residuos
    results = []
    residuals = []

    for name, group in filtered_data.groupby('EquipmentName'):
        X = group['ParameterFloatValue_x'].values.reshape(-1, 1)
        y = group['ParameterFloatValue_y'].values
        
        if name in models:
            lin_reg, poly_reg = models[name]
            y_pred = lin_reg.predict(poly_reg.transform(X))
            
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            
            results.append({
                "EquipmentName": name,
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae
            })
            
            # Calcular residuos
            res = y - y_pred
            residuals.extend([{
                "EquipmentName": name,
                "ReadTime": rt,
                "Residual": r,
                "Real": real,
                "Predicted": pred
            } for rt, real, pred, r in zip(group['ReadTime'], y, y_pred, res)])

    # Convertir los residuos en un DataFrame
    residuals_df = pd.DataFrame(residuals)

    # Graficar residuos por EquipmentName
    st.write("### Gráfico de Residuos por CAEX >800 RPM")

    if not residuals_df.empty:
        nrows = int(np.ceil(np.sqrt(len(residuals_df['EquipmentName'].unique()))))
        ncols = int(np.ceil(len(residuals_df['EquipmentName'].unique()) / nrows))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 8))

        for (name, group), ax in zip(residuals_df.groupby('EquipmentName'), axes.flat):
            ax.scatter(group['ReadTime'], group['Residual'], color='red', alpha=0.5, s=10, label='Residual')
            ax.scatter(group['ReadTime'], group['Real'], color='blue', alpha=0.5, s=10, label='Real')
            ax.scatter(group['ReadTime'], group['Predicted'], color='green', alpha=0.5, s=10, label='Predicted')
            ax.set_title(f"{name}", fontsize=8)
            ax.set_xlabel('ReadTime', fontsize=8)
            ax.set_ylabel('Values', fontsize=8)
            ax.legend(fontsize=5)
            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=6)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("No hay datos de residuos disponibles.")
