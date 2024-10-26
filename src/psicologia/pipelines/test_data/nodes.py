import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTEENN

def generar_datos_prueba(data: pd.DataFrame):
    # Convertir columnas categóricas a numéricas usando Label Encoding
    label_encoder = LabelEncoder()
    categorical_columns = ['Timestamp', 'Gender', 'Country', 'Occupation', 'Days_Indoors', 'Mood_Swings']
    
    for col in categorical_columns:
        data[col] = label_encoder.fit_transform(data[col])

    # Definir características y objetivo
    features = data.drop('treatment', axis=1)
    target = data['treatment']

    # Usar StratifiedShuffleSplit para asegurar distribución representativa de clases
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in stratified_split.split(features, target):
        X_test = features.iloc[test_idx]
        y_test = target.iloc[test_idx]

    # Aplicar SMOTEENN para balancear las clases en el conjunto de prueba
    smote_enn = SMOTEENN(random_state=42)
    X_test_balanced, y_test_balanced = smote_enn.fit_resample(X_test, y_test)

    return X_test_balanced, y_test_balanced
