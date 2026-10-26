# ============================================================
# K-Means + PCA sobre DATASET_Inmuebles.xlsx
# ============================================================
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

# 1. Cargar dataset
df = pd.read_excel("DATASET_Inmuebles.xlsx")
df = df.rename(columns={
    "Inmueble": "ID_inmueble", "Área m²": "Area_m2", "Precio arriendo": "Precio_arriendo",
    "Baños": "Banos", "Antig[uedad": "Antiguedad_anios", "Tipo Inmueble": "Tipo_inmueble"})
df = df.drop(columns=["Estrato.1"])           # segunda columna "Estrato" (no descrita en el enunciado)
ids = df["ID_inmueble"]                        # solo para identificar registros

variables_numericas = ["Precio_arriendo", "Area_m2", "Habitaciones", "Banos", "Antiguedad_anios", "Estrato"]
variables_categoricas = ["Localidad", "Tipo_inmueble"]
X = df[variables_numericas + variables_categoricas]
print("Dimensiones originales (sin ID):", X.shape)

# 2. Preprocesamiento
prep = ColumnTransformer([
    ("numericas", StandardScaler(), variables_numericas),
    ("categoricas", OneHotEncoder(), variables_categoricas)])
Xp = prep.fit_transform(X)
print("Dimensiones después del preprocesamiento:", Xp.shape)

# 3. Experimento A: K-Means sobre variables preprocesadas
K = 3
kmA = KMeans(n_clusters=K, random_state=42, n_init=10)
gA = kmA.fit_predict(Xp)
df["Grupo_KMeans"] = gA

# 4. PCA
pca = PCA(n_components=2)
Xpca = pca.fit_transform(Xp)
print("Dimensiones después de PCA:", Xpca.shape)
print("Varianza explicada:", pca.explained_variance_ratio_, "acumulada:", pca.explained_variance_ratio_.sum())
pca_full = PCA().fit(Xp)
print("Varianza acumulada (todos):", np.cumsum(pca_full.explained_variance_ratio_).round(3))
nombres = prep.get_feature_names_out()
cargas = pd.DataFrame(pca.components_.T, index=nombres, columns=["PC1", "PC2"])
print("Cargas:\n", cargas.round(3))

# 5. Gráfica de los grupos en PC1-PC2
plt.figure(figsize=(6.5, 4.8))
for g in sorted(df["Grupo_KMeans"].unique()):
    s = gA == g
    plt.scatter(Xpca[s, 0], Xpca[s, 1], label=f"Grupo {g}", alpha=0.7, s=14)
plt.xlabel("Componente principal 1"); plt.ylabel("Componente principal 2")
plt.title("Segmentación inmobiliaria: K-Means + PCA"); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig("pca_kmeans.png", dpi=150); plt.close()

# Varianza acumulada
plt.figure(figsize=(6, 3.8))
plt.bar(range(1, len(pca_full.explained_variance_ratio_)+1), pca_full.explained_variance_ratio_, label="Por componente")
plt.plot(range(1, len(pca_full.explained_variance_ratio_)+1), np.cumsum(pca_full.explained_variance_ratio_), "o-", color="k", label="Acumulada")
plt.xlabel("Componente"); plt.ylabel("Varianza explicada"); plt.legend(); plt.grid(True, alpha=.4)
plt.tight_layout(); plt.savefig("pca_varianza.png", dpi=150); plt.close()

# 6. Perfil comercial
print("\nTamaños:\n", df["Grupo_KMeans"].value_counts().sort_index())
print(df.groupby("Grupo_KMeans")[variables_numericas].mean().round(2))
print(pd.crosstab(df["Grupo_KMeans"], df["Localidad"], normalize="index").round(2))
print(pd.crosstab(df["Grupo_KMeans"], df["Tipo_inmueble"], normalize="index").round(2))

# 7. Experimento B: K-Means sobre componentes principales
print("\n--- Experimento B ---")
for nc in (2, 3, 4, 5, 6, 8):
    Z = PCA(n_components=nc).fit_transform(Xp)
    gB = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(Z)
    print(f"PCA {nc} comp (var {PCA(n_components=nc).fit(Xp).explained_variance_ratio_.sum():.3f}): ARI vs A = {adjusted_rand_score(gA, gB):.3f}, coincidencia directa = {(gA==gB).mean():.3f}")
gB2 = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(Xpca)
df["Grupo_KMeans_PCA2"] = gB2
print("Tabla cruzada A vs B(2 comp):\n", pd.crosstab(df["Grupo_KMeans"], df["Grupo_KMeans_PCA2"]))
print("Silhouette A (espacio preprocesado):", round(silhouette_score(Xp, gA), 3))
print("Silhouette B (espacio PCA2):", round(silhouette_score(Xpca, gB2), 3))
print(df.groupby("Grupo_KMeans_PCA2")[variables_numericas].mean().round(2))
for k in (2, 3, 4):
    g = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xp)
    print(f"k={k} sobre preprocesado: silhouette={silhouette_score(Xp, g):.3f}")
# gráfica B
plt.figure(figsize=(6.5, 4.8))
for g in sorted(set(gB2)):
    s = gB2 == g
    plt.scatter(Xpca[s, 0], Xpca[s, 1], label=f"Grupo {g}", alpha=0.7, s=14)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("K-Means aplicado sobre PC1-PC2"); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig("pca_kmeans_B.png", dpi=150); plt.close()
