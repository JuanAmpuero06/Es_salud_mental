from kedro.pipeline import Pipeline, node
from .nodes import *

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=cargar_datos_del_catalogo,
                inputs="Mental_Health_Dataset",
                outputs="datos_salud_mental",
                name="nodo_cargar_datos",
            ),
            node(
                func=verificar_columnas,  # Nuevo nodo para revisar las columnas
                inputs="datos_salud_mental",
                outputs="datos_verificados",
                name="nodo_verificar_columnas",
            ),
            node(
                func=analizar_datos,
                inputs="datos_verificados",  # Cambia la entrada a "datos_verificados"
                outputs="resumen_datos",
                name="nodo_analizar_datos",
            ),
            node(
                func=verificar_valores_faltantes,
                inputs="datos_verificados",  # Cambia la entrada a "datos_verificados"
                outputs="resumen_valores_faltantes",
                name="nodo_verificar_valores_faltantes",
            ),
            node(
                func=graficar_correlaciones,
                inputs="datos_verificados",  # Cambia la entrada a "datos_verificados"
                outputs=None,
                name="nodo_graficar_correlaciones",
            ),
        ]
    )
