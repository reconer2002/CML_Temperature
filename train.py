import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# ============================
# 1. Cargar datos
# ============================
data = pd.read_csv("data.csv")
data = data.rename(columns={data.columns[0]: "id"})
print("Columnas detectadas:", data.columns)

# ============================
# 2. Preprocesamiento
# ============================
data["Date"] = pd.to_datetime(data["Date"], format="%d.%m.%Y")
data["Date"] = data["Date"].astype("int64") // 10**9  # timestamp en segundos

# Codificar categorías
le_weather = LabelEncoder()
le_cloud = LabelEncoder()
data["weather"] = le_weather.fit_transform(data["weather"].astype(str))
data["cloud"] = le_cloud.fit_transform(data["cloud"].astype(str))

# ============================
# 3. Definir X e y
# ============================
target = "maxtemp"
X = data.drop(columns=["maxtemp", "id"])
y = data[target]

# ============================
# 4. División y escalado
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================
# 5. Optimización de hiperparámetros para maximizar R²
# ============================
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 12, 18, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='r2',      # Maximizar R²
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print("\nMejores hiperparámetros:", grid_search.best_params_)
print("Mejor R² en CV:", grid_search.best_score_)

# ============================
# 6. Predicciones
# ============================
y_pred = best_model.predict(X_test_scaled)

# ============================
# 7. Métricas
# ============================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nMAE: {mae:.3f}")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")

with open("metrics.txt", "w") as f:
    f.write(f"MAE: {mae:.3f}\n")
    f.write(f"MSE: {mse:.3f}\n")
    f.write(f"RMSE: {rmse:.3f}\n")
    f.write(f"R2: {r2:.3f}\n")

# ============================
# 8. Guardar modelo y objetos
# ============================
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_weather, "label_weather.pkl")
joblib.dump(le_cloud, "label_cloud.pkl")

# ============================
# 9. Visualización
# ============================
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Real")
plt.plot(y_pred, label="Predicción", linestyle="--")
plt.title("Predicción de Temperatura Máxima")
plt.xlabel("Muestras")
plt.ylabel("Temperatura")
plt.legend()
plt.savefig("pred_vs_real.png")
plt.close()

print("\nArchivos generados:")
print(" - model.pkl")
print(" - scaler.pkl")
print(" - label_weather.pkl")
print(" - label_cloud.pkl")
print(" - pred_vs_real.png")
print(" - metrics.txt")
