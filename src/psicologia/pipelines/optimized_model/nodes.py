import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTEENN
import pandas as pd
import os

def entrenar_modelo():
    # Cargar el dataset desde la ruta proporcionada
    tpot_data = pd.read_csv('data/02_intermediate/mapped_data.csv')

    # Eliminar columnas irrelevantes como 'Timestamp', 'Country' y 'Occupation'
    tpot_data = tpot_data.drop(['Timestamp', 'Country', 'Occupation'], axis=1)

    # Asegurarse de que la columna objetivo sea 'treatment'
    features = tpot_data.drop('treatment', axis=1)
    target = tpot_data['treatment']

    # Codificar columnas categóricas
    label_encoder = LabelEncoder()
    for column in features.select_dtypes(include=['object']).columns:
        features[column] = label_encoder.fit_transform(features[column])

    # Aplicar SMOTEENN para balancear las clases
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(features, target)

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Guardar X_test y y_test en archivos CSV
    os.makedirs('data/05_model_input/', exist_ok=True)  # Crear directorio si no existe
    pd.DataFrame(X_test).to_csv('data/05_model_input/X_test.csv', index=False)
    pd.DataFrame(y_test).to_csv('data/05_model_input/y_test.csv', index=False)

    # Configurar los hiperparámetros para RandomizedSearchCV
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.05, 0.1],  # Valores más bajos para ajustes más finos
        'max_depth': [3, 4, 5, 6, 7, 8, 10],  # Profundidades más variadas
        'min_child_weight': [1, 3, 5, 7],  # Más opciones para ajustar
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],  # Más valores para la muestra
        'n_estimators': [200, 300, 500, 1000],  # Más árboles para mayor tiempo de entrenamiento
        'colsample_bytree': [0.6, 0.8, 1.0],  # Diversas fracciones de columnas usadas
        'gamma': [0, 0.1, 0.2, 0.3, 0.5],  # Variación en la penalización de particiones
        'scale_pos_weight': [1, 2, 3, 5, 10]  # Más opciones para ajustar desequilibrio
    }

    # Crear el modelo XGBClassifier
    xgb_model = XGBClassifier(random_state=42)

    # Configurar la validación cruzada
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # 10 pliegues para mayor estabilidad

    # Configurar RandomizedSearchCV con más iteraciones
    random_search = RandomizedSearchCV(estimator=xgb_model, 
                                       param_distributions=param_grid, 
                                       n_iter=100,  # Aumentado a 100 iteraciones
                                       cv=cv, 
                                       n_jobs=-1, 
                                       verbose=2, 
                                       random_state=42, 
                                       scoring='accuracy')

    # Entrenar el modelo
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    # Crear el directorio 'data/06_models/' si no existe
    os.makedirs('data/06_models/', exist_ok=True)

    # Guardar el mejor modelo optimizado
    joblib.dump(best_model, 'data/06_models/modelo_optimizado.pkl')

    return best_model, X_test, y_test
