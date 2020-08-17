![png](../../../imagenes/logotipo-axity-ppt.png)

# Linear Discriminant Analysis (LDA)

## Explicación del Ejemplo

El propósito de este ejemplo es identificar si una imagen contiene un rostro humano. Se utilizará LDA, para reducir el número de características a utilizar.

### Los datos
Se utilizará el conjunto de datos fetch_olivetti_faces de sklearn.  

El conjunto contiene imágenes de rostros humanos, tomadas entre abril 1992 y abril 1994, en los laboratorios AT&T de Cambridge.  
Hay 10 imágenes diferentes de 40 personas distintas. Para algunas personas, las imágenes fueron tomadas en tienmpos diferentes, variando las condiciones de luz, expresiones faciales y detalles faciales (por ejemplo anteojos).  
Son imágenes de 64x64 pixeles y se ajustaron a 256 niveles de gris. Están almacenadas como enteros de 8-bits sin signo.  
Las imágenes tienen una etiqueta entre 0 y 39, indicando la identidad de la persona.  


```python
# Importemos las librerias
import matplotlib.pyplot as plt
import numpy as np
```

## Examinar los datos


```python
# Importemos el conjunto de datos
from sklearn import datasets
faces = datasets.fetch_olivetti_faces()
faces.data.shape
```




    (400, 4096)




```python
# Visualicemos los datos
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(8, 6))
# plot several images
for i in range(15):
    ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(faces.images[i], cmap=plt.cm.bone)
```


![png](../../../imagenes/03-Linear%20Discriminant%20Analysis-Solucion_5_0.png)


## LDA


```python
# Dividir el conjunto de datos en conjunto de entrenamiento y conjunto de prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(faces.data,
        faces.target, random_state=0)

print(X_train.shape, X_test.shape)
```

    (300, 4096) (100, 4096)
    


```python
# Reducir la dimensión del dataset con LDA
# Nota: llamar LDA sin parámetros
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA()
lda.fit_transform(X_train, y_train)
```




    array([[ 9.1825219 ,  4.89328857, -0.67963778, ...,  1.07263403,
            -0.95794836, -1.124833  ],
           [ 4.26019219,  0.33021323,  1.99890265, ..., -0.10164646,
            -0.56319177,  0.62339041],
           [ 5.6931902 ,  6.1481827 ,  8.53287892, ..., -0.32646258,
            -1.18561162, -0.82515512],
           ...,
           [ 3.60473467, -6.60268835, -0.48627446, ..., -0.88605172,
             3.5307394 , -0.09106856],
           [ 3.60125454, -8.4345999 ,  2.90119703, ..., -0.36022016,
            -0.08654677,  0.20317931],
           [ 0.34369362, -5.92805471, -1.10379853, ...,  0.32389482,
            -1.50949595,  1.37672824]])




```python
# Apliquemos LDA a nuestros conjuntos de entrenamiento y prueba
X_train_lda = lda.transform(X_train)
X_test_lda = lda.transform(X_test)
print(X_train_lda.shape)
print(X_test_lda.shape)
```

    (300, 39)
    (100, 39)
    

## Entrenar el modelo


```python
# Ajustar el modelo de SVM en el Conjunto de Entrenamiento
from sklearn import svm
clf = svm.SVC(C=5., gamma=0.001)
clf.fit(X_train_lda, y_train)
```




    SVC(C=5.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)




```python
# Predicción de los resultados con el Conjunto de Prueba
fig = plt.figure(figsize=(8, 6))
for i in range(15):
    ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(X_test[i].reshape(faces.images[0].shape),
              cmap=plt.cm.bone)
    y_pred = clf.predict(X_test_lda[i, np.newaxis])[0]
    color = ('black' if y_pred == y_test[i] else 'red')
    ax.set_title(y_pred, fontsize='small', color=color)

y_pred = clf.predict(X_test_lda)
```


![png](../../../imagenes/03-Linear%20Discriminant%20Analysis-Solucion_12_0.png)



```python
# Elaborar una matriz de confusión
from sklearn import metrics
print(metrics.confusion_matrix(y_test, y_pred))
```

    [[6 0 0 ... 0 0 0]
     [0 4 0 ... 0 0 0]
     [0 0 2 ... 0 0 0]
     ...
     [0 0 0 ... 3 0 0]
     [0 0 0 ... 0 1 0]
     [0 0 0 ... 0 0 3]]
    


```python
# Obtener el informe de clasificación
print(metrics.classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00         6
               1       1.00      1.00      1.00         4
               2       0.67      1.00      0.80         2
               3       1.00      1.00      1.00         1
               4       1.00      1.00      1.00         1
               5       1.00      1.00      1.00         5
               6       1.00      1.00      1.00         4
               7       1.00      1.00      1.00         3
               9       1.00      1.00      1.00         1
              10       1.00      1.00      1.00         4
              11       1.00      1.00      1.00         1
              12       1.00      1.00      1.00         2
              13       1.00      1.00      1.00         3
              14       1.00      1.00      1.00         5
              15       1.00      1.00      1.00         3
              17       1.00      1.00      1.00         6
              19       1.00      1.00      1.00         4
              20       1.00      1.00      1.00         1
              21       1.00      1.00      1.00         1
              22       1.00      1.00      1.00         2
              23       1.00      1.00      1.00         1
              24       1.00      1.00      1.00         2
              25       1.00      0.50      0.67         2
              26       1.00      1.00      1.00         4
              27       1.00      1.00      1.00         1
              28       1.00      1.00      1.00         2
              29       1.00      1.00      1.00         3
              30       1.00      1.00      1.00         4
              31       1.00      1.00      1.00         3
              32       1.00      1.00      1.00         3
              33       1.00      1.00      1.00         2
              34       1.00      1.00      1.00         3
              35       1.00      1.00      1.00         1
              36       1.00      1.00      1.00         3
              37       1.00      1.00      1.00         3
              38       1.00      1.00      1.00         1
              39       1.00      1.00      1.00         3
    
        accuracy                           0.99       100
       macro avg       0.99      0.99      0.99       100
    weighted avg       0.99      0.99      0.99       100
    
    
