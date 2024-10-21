from kedro.pipeline import Pipeline, node
from .nodes import predecir_con_modelo


def create_pipeline(**kwargs):
    return Pipeline([
        node(predecir_con_modelo, inputs=None, outputs="y_pred")
    ])