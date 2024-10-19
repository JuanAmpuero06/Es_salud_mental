from kedro.pipeline import Pipeline, node
from .nodes import entrenar_modelo

def create_pipeline(**kwargs):
    return Pipeline([
        node(entrenar_modelo, inputs=None, outputs=["modelo_optimizado", "X_test", "y_test"])
    ])
