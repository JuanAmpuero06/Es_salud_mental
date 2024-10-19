import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

# Cargar el dataset desde la ruta proporcionada
tpot_data = pd.read_csv(r'C:\Users\jampu\OneDrive\Desktop\salud_mental\psicologia\notebooks\mapped_data.csv')

# Eliminar columnas irrelevantes como 'Timestamp'
tpot_data = tpot_data.drop(['Timestamp'], axis=1)

# Asegurarse de que la columna objetivo sea 'treatment'
features = tpot_data.drop('treatment', axis=1)
target = tpot_data['treatment']

# Codificar columnas categóricas
label_encoder = LabelEncoder()
for column in features.select_dtypes(include=['object']).columns:
    features[column] = label_encoder.fit_transform(features[column])

# Aplicar SMOTEENN para sobremuestreo y submuestreo combinado
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(features, target)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Configuración de la búsqueda de hiperparámetros con RandomizedSearchCV
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [4, 6, 8],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.8, 1.0],
    'n_estimators': [200, 300],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.3]
}

# Crear el modelo XGBClassifier
xgb_model = XGBClassifier(random_state=42)

# Configurar la validación cruzada
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Configurar RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid, 
                                   n_iter=10, cv=cv, n_jobs=-1, verbose=2, random_state=42, scoring='accuracy')

# Medir el tiempo de inicio
start_time = time.time()

# Entrenar con la búsqueda de hiperparámetros
random_search.fit(X_train, y_train)

# Mostrar el tiempo total de ejecución
print(f"Tiempo de ejecución: {time.time() - start_time} segundos")

# Mostrar los mejores hiperparámetros encontrados
print(f"Mejores hiperparámetros: {random_search.best_params_}")

# Entrenar el modelo con los mejores hiperparámetros
best_model = random_search.best_estimator_

# Hacer predicciones en el conjunto de prueba
y_pred = best_model.predict(X_test)

# Evaluar el rendimiento
print(f"Accuracy del modelo optimizado: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))
