import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def cargar_datos_del_catalogo(data: pd.DataFrame):
    # Elimina la columna "Timestamp" si existe
    if "Timestamp" in data.columns:
        data = data.drop(columns=["Timestamp"])
    return data

def analizar_datos(data: pd.DataFrame):
    resumen = data.describe()
    return resumen

def verificar_valores_faltantes(data: pd.DataFrame):
    faltantes = data.isnull().sum()
    return faltantes

def graficar_correlaciones(data: pd.DataFrame):
    # Elimina columnas no numéricas
    datos_numericos = data.select_dtypes(include=[float, int])
    # Verifica si hay columnas numéricas después de la selección
    if datos_numericos.empty:
        print("No hay columnas numéricas disponibles para calcular correlaciones.")
        return
    # Calcula las correlaciones
    correlaciones = datos_numericos.corr()
    # Verifica si hay valores en la matriz de correlación
    if correlaciones.empty:
        print("La matriz de correlación está vacía, no se puede generar el heatmap.")
        return
    # Genera un heatmap de las correlaciones
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlaciones, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de Correlación")
    plt.show()

def verificar_columnas(data: pd.DataFrame):
    print("Columnas después de la limpieza:", data.columns)
    return data
