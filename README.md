# Segmentación de inmuebles en arriendo con K-Means y PCA

Trabajo de la asignatura Introducción a Machine Learning, ESP. Inteligencia Artificial, Universidad de Cundinamarca (2026-2).

Autor: Kevin Esteban Chiquillo Díaz
Docente: Ing. Wilson Rojas Reales

## Descripción

Una inmobiliaria quiere saber si sus inmuebles en arriendo se pueden agrupar de forma natural según su precio, área, habitaciones, baños, antigüedad, estrato, localidad y tipo. Para eso se usan dos técnicas de aprendizaje no supervisado:

- **K-Means**, para encontrar los grupos de inmuebles parecidos.
- **PCA**, para reducir los datos a 2 dimensiones y poder verlos en un gráfico.


## Archivos

| Archivo | Qué hace |
|---|---|
| `KMeansSegInmuebles_Dataset.py` | Aplica K-Means con 2, 3 y 4 clusters, calcula inercia y silueta, y genera una gráfica por cada k. |
| `Algoritmo_InmueblesPCA_Dataset.py` | Aplica K-Means (k = 3) y PCA, muestra la varianza explicada y compara agrupar con las variables completas frente a agrupar con los componentes principales. |
