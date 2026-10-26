# ============================================================
# LABORATORIO: Segmentación de inmuebles en arriendo (K-Means)
# Versión modificada para leer DATASET_Inmuebles.xlsx y probar k = 2, 3 y 4
# ============================================================
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Leer dataset (la hoja trae dos columnas llamadas "Estrato"; se usa la primera,
#    que es la descrita en el enunciado, y se descarta la segunda: "Estrato.1")
df = pd.read_excel("DATASET_Inmuebles.xlsx")
df = df.rename(columns={
    "Inmueble": "ID_inmueble", "Área m²": "Area_m2", "Precio arriendo": "Precio_arriendo",
    "Baños": "Banos", "Antig[uedad": "Antiguedad_anios", "Tipo Inmueble": "Tipo_inmueble"})
df = df.drop(columns=["Estrato.1"])
print("Dimensiones:", df.shape)

# 2. Variables numéricas (el ID NO se usa para calcular distancias)
vars_num = ["Precio_arriendo", "Area_m2", "Habitaciones", "Banos", "Antiguedad_anios", "Estrato"]
X = df[vars_num]

# 3. Escalamiento
X_scaled = StandardScaler().fit_transform(X)
scaler = StandardScaler().fit(X)

# 4. K-Means con k = 2, 3 y 4
resultados = []
for k in (2, 3, 4):
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    modelo.fit(X_scaled)
    col = f"cluster_k{k}"
    df[col] = modelo.labels_
    sil = silhouette_score(X_scaled, modelo.labels_)
    resultados.append((k, modelo.inertia_, sil))
    print(f"\n===== k = {k} =====  inercia={modelo.inertia_:.1f}  silhouette={sil:.3f}")
    print("Tamaño por cluster:\n", df[col].value_counts().sort_index())
    cent = pd.DataFrame(scaler.inverse_transform(modelo.cluster_centers_), columns=vars_num)
    print("Centroides (escala original):\n", cent.round(2))
    print("Resumen promedio por cluster:\n", df.groupby(col)[vars_num].mean().round(2))
    print("Localidad (proporción):\n", pd.crosstab(df[col], df["Localidad"], normalize="index").round(2))
    print("Tipo inmueble (proporción):\n", pd.crosstab(df[col], df["Tipo_inmueble"], normalize="index").round(2))
    plt.figure(figsize=(6, 4.5))
    plt.scatter(df["Area_m2"], df["Precio_arriendo"], c=df[col], cmap="viridis", s=14, alpha=0.7)
    plt.xlabel("Área del inmueble (m²)"); plt.ylabel("Precio de arriendo (COP)")
    plt.title(f"Segmentación de inmuebles en arriendo (k={k})")
    plt.tight_layout(); plt.savefig(f"kmeans_k{k}.png", dpi=150); plt.close()

print("\nk, inercia, silhouette:", resultados)
df.to_excel("inmuebles_con_clusters.xlsx", index=False)
