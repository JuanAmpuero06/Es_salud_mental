from kedro.pipeline import Pipeline, node
from .nodes import *

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(detectar_nulos, inputs="Mental_Health_Dataset", outputs="nulos_data", name="detectar_nulos"),
            node(imputar_self_employed, inputs="Mental_Health_Dataset", outputs="imputed_data", name="imputar_self_employed"),
            node(mapear_columnas, inputs="imputed_data", outputs="mapped_data", name="mapear_columnas"),
        ]
    )
