import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def predecir_con_modelo():
    # Cargar el modelo optimizado desde el archivo
    modelo_optimizado = joblib.load('data/06_models/modelo_optimizado.pkl')

    # Cargar los datos de prueba X_test y y_test desde los archivos CSV
    X_test = pd.read_csv('data/05_model_input/X_test.csv')
    y_test = pd.read_csv('data/05_model_input/y_test.csv')

    # Hacer predicciones
    y_pred = modelo_optimizado.predict(X_test)

    # Evaluar el rendimiento del modelo
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Mostrar los resultados
    print(f"Accuracy del modelo optimizado: {accuracy * 100:.2f}%")
    print("Matriz de Confusión:")
    print(conf_matrix)
    print("Reporte de Clasificación:")
    print(class_report)

    # Retornar las predicciones
    return y_pred