import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def detectar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    nulos_por_columna = df.isnull().sum()
    df_nulos = pd.DataFrame(nulos_por_columna, columns=['Valores Nulos'])
    df_nulos['Porcentaje Nulos'] = (nulos_por_columna / len(df)) * 100
    return df_nulos

def imputar_self_employed(df: pd.DataFrame) -> pd.DataFrame:
    df_train = df[df['self_employed'].notnull()]
    df_null = df[df['self_employed'].isnull()]

    features = ['Growing_Stress', 'Mental_Health_History']
    X_train = df_train[features]
    y_train = df_train['self_employed']
    X_null = df_null[features]

    # Codificación de las columnas categóricas
    le_self_employed = LabelEncoder()
    le_stress = LabelEncoder()
    le_mental_health = LabelEncoder()

    X_train_encoded = X_train.copy()
    X_train_encoded['Growing_Stress'] = le_stress.fit_transform(X_train['Growing_Stress'])
    X_train_encoded['Mental_Health_History'] = le_mental_health.fit_transform(X_train['Mental_Health_History'])
    y_train_encoded = le_self_employed.fit_transform(y_train)

    X_null_encoded = X_null.copy()
    X_null_encoded['Growing_Stress'] = le_stress.transform(X_null['Growing_Stress'])
    X_null_encoded['Mental_Health_History'] = le_mental_health.transform(X_null['Mental_Health_History'])

    # Entrenamiento del modelo
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_encoded, y_train_encoded)

    # Predicción de valores nulos
    y_pred_encoded = model.predict(X_null_encoded)
    y_pred = le_self_employed.inverse_transform(y_pred_encoded)

    # Imputar los valores nulos
    df.loc[df['self_employed'].isnull(), 'self_employed'] = y_pred

    return df

def mapear_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df['self_employed'] = df['self_employed'].map({'Yes': 1, 'No': 0})
    df['family_history'] = df['family_history'].map({'Yes': 1, 'No': 0})
    df['treatment'] = df['treatment'].map({'Yes': 1, 'No': 0})
    df['Coping_Struggles'] = df['Coping_Struggles'].map({'Yes': 1, 'No': 0})

    mapping = {'No': 0, 'Yes': 1, 'Maybe': 2}
    df['Growing_Stress'] = df['Growing_Stress'].map(mapping)
    df['Changes_Habits'] = df['Changes_Habits'].map(mapping)
    df['Mental_Health_History'] = df['Mental_Health_History'].map(mapping)
    df['Work_Interest'] = df['Work_Interest'].map(mapping)
    df['Social_Weakness'] = df['Social_Weakness'].map(mapping)
    df['mental_health_interview'] = df['mental_health_interview'].map(mapping)

    mapping_care_options = {'No': 0, 'Yes': 1, 'Not sure': 2}
    df['care_options'] = df['care_options'].map(mapping_care_options)

    return df

