from kedro.pipeline import Pipeline, node
from .nodes import generar_datos_prueba

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            generar_datos_prueba,
            inputs="mapped_data",  # Usa el dataset procesado de entrada
            outputs=["X_test_balanced", "y_test_balanced"],
            name="generar_datos_prueba"
        )
    ])
