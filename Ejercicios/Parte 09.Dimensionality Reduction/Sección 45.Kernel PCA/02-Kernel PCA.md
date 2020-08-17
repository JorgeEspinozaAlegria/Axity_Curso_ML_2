![png](../../../imagenes/logotipo-axity-ppt.png)

# Kernel PCA

## Explicación del Ejemplo

Se verán varios ejemplos, comparando PCA vs. Kernel PCA, con datos generados artificialmente.  

**Nota: Este ejercicio no utiliza ningún algoritmo de ML. El propósito es mostrar las diferencias entre PCA y Kernel PCA**  


```python
# Importar las librerias
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
```

## Ejemplo 1


```python
# Generar los datos
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, random_state=123)

plt.figure(figsize=(8,6))

plt.scatter(X[y==0, 0], X[y==0, 1], color='red', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', alpha=0.5)

plt.title('Un conjunto de datos no lineal')
plt.ylabel('coordenadas y')
plt.xlabel('coordenadas x')

plt.show()
```


![png](../../../imagenes/02-Kernel%20PCA_4_0.png)


### PCA


```python
# Veamos la reducción de dimensiones con PCA
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_spca[y==0, 0], X_spca[y==0, 1], color='red', alpha=0.5)
plt.scatter(X_spca[y==1, 0], X_spca[y==1, 1], color='blue', alpha=0.5)

plt.title('Primeros 2 componentes principales después de PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
```


![png](../../../imagenes/02-Kernel%20PCA_6_0.png)



```python
scikit_pca = PCA(n_components=1)
X_spca = scikit_pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_spca[y==0, 0], np.zeros((50,1)), color='red', alpha=0.5)
plt.scatter(X_spca[y==1, 0], np.zeros((50,1)), color='blue', alpha=0.5)

plt.title('Primer componente principal después de PCA')
plt.xlabel('PC1')

plt.show()
```


![png](../../../imagenes/02-Kernel%20PCA_7_0.png)


Como podemos observar, los componentes principales no generan un subespacio en donde los datos puedan ser separados de manera lineal.  
PCA es un método no supervisado y no considera las etiquetas, para maximizar la variación entre las clases. En estas gráficas, se usan los colores para facilitar la visualización de las 2 clases.

### Kernel PCA


```python
# Veamos la reducción de dimensiones con Kernel PCA
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1], color='red', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1], color='blue', alpha=0.5)

plt.title('Primeros 2 componentes principales después de Kernel PCA (RBF)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
```


![png](../../../imagenes/02-Kernel%20PCA_10_0.png)



```python
scikit_kpca = KernelPCA(n_components=1, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_skernpca[y==0, 0], np.zeros((50,1)), color='red', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], np.zeros((50,1)), color='blue', alpha=0.5)

plt.title('Primer componente principal después de Kernel PCA (RBF)')
plt.xlabel('PC1')
plt.show()
```


![png](../../../imagenes/02-Kernel%20PCA_11_0.png)


## Ejemplo 2


```python
# Generar los datos
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

plt.figure(figsize=(8,6))

plt.scatter(X[y==0, 0], X[y==0, 1], color='red', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', alpha=0.5)

plt.title('Concentric circles')
plt.ylabel('y coordinate')
plt.xlabel('x coordinate')
plt.show()
```


![png](../../../imagenes/02-Kernel%20PCA_13_0.png)


### PCA


```python
# Veamos la reducción de dimensiones con PCA
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X[y==0, 0], np.zeros((500,1))+0.1, color='red', alpha=0.5)
plt.scatter(X[y==1, 0], np.zeros((500,1))-0.1, color='blue', alpha=0.5)
plt.ylim([-15,15])

plt.title('Primer componente principal después de PCA')
plt.xlabel('PC1')
plt.show()
```


![png](../../../imagenes/02-Kernel%20PCA_15_0.png)


### Kernel PCA


```python
# Veamos la reducción de dimensiones con Kernel PCA
scikit_kpca = KernelPCA(n_components=1, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_skernpca[y==0, 0], np.zeros((500,1)), color='red', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], np.zeros((500,1)), color='blue', alpha=0.5)

plt.title('Primer componente principal después de Kernel PCA (RBF)')
plt.xlabel('PC1')
plt.show()
```


![png](../../../imagenes/02-Kernel%20PCA_17_0.png)

