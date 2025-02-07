import numpy as np
import matplotlib.pyplot as plt

def kmeans_from_scratch(X, k, n_init=10, max_iter=300, tol=1e-4, random_state=42):
    np.random.seed(random_state)
    n_points, n_features = X.shape
    best_inertia = np.inf
    best_labels = None
    best_centroids = None

    # Se realizan n_init inicializaciones y se elige la que tenga menor inercia.
    for init in range(n_init):
        # Seleccionar k puntos aleatorios como centroides iniciales (sin reemplazo)
        indices = np.random.choice(n_points, k, replace=False)
        centroids = X[indices].copy()

        for iteration in range(max_iter):
            # Calcular la distancia de cada punto a cada centroide (distancia Euclidiana)
            distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            # Calcular nuevos centroides como la media de los puntos asignados a cada cluster
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                if np.any(labels == j):
                    new_centroids[j] = X[labels == j].mean(axis=0)
                else:
                    # Si un cluster se queda sin puntos, se re-inicializa ese centroide aleatoriamente
                    new_centroids[j] = X[np.random.choice(n_points)]
            
            # Verificar la convergencia: si todos los centroides cambian menos que la tolerancia
            if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
                centroids = new_centroids
                break

            centroids = new_centroids

        # Calcular la inercia (suma de las distancias al cuadrado de cada punto a su centroide)
        inertia = np.sum((np.linalg.norm(X - centroids[labels], axis=1)) ** 2)
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()

    return best_labels, best_centroids, best_inertia

# ------------------- Uso del algoritmo K-Means desde cero -------------------

# Cargar datos reducidos
features_2d = np.load("data/reduction/test_tsne_2d.npy")

#parmetros
k = 5
n_init = 10
max_iter = 300
tol = 1e-4
random_state = 42

labels, centroids, inertia = kmeans_from_scratch(features_2d, k, n_init, max_iter, tol, random_state)

# Guardar etiquetas
np.save("data/reduction/test_kmeans_labels.npy", labels)

print("✅ Clustering con K-Means implementado desde cero completado y guardado.")

"""
# Opcional: Visualización de los clusters y los centroides
plt.figure(figsize=(8, 6))
unique_labels = np.unique(labels)
colors = plt.cm.get_cmap("viridis", len(unique_labels))

for idx, label in enumerate(unique_labels):
    plt.scatter(features_2d[labels == label, 0],
                features_2d[labels == label, 1],
                s=30, color=colors(idx), label=f"Cluster {label}")

# Dibujar los centroides
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label="Centroides")
plt.title("Clustering con K-Means (Implementación desde Cero)")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.legend()
plt.show()
"""