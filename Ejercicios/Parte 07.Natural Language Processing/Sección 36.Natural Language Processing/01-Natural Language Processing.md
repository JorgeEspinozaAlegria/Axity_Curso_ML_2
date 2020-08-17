![png](../../../imagenes/logotipo-axity-ppt.png)

# Procesamiento de Lenguaje Natural

## Explicación del Ejemplo

El propósito de este ejemplo es construir un modelo, que prediga si una opinión proporcionada por un cliente, sobre un restaurante, es positiva o negativa.  

### Los datos
Se utilizará el conjunto de datos Restaurant Reviews de Kaggle.  
Este consiste en 1000 opiniones de comensales sobre un restaurante y si fue positiva o negativa.  

Debido a que contamos con datos etiquetados, podemos utilizar algoritmos de aprendizaje supervisados y obtener la precisión del modelo


```python
# Importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```


```python
# Importar el conjunto de datos
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)
```


```python
# Revisión del conjunto de datos
dataset.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Review</th>
      <th>Liked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wow... Loved this place.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Crust is not good.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Not tasty and the texture was just nasty.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Stopped by during the late May bank holiday of...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The selection on the menu was great and so wer...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Limpieza de texto
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
```



```python
# Crear el Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
```


```python
# Dividir el conjunto de datos en conjunto de entrenamiento y conjunto de prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
```


```python
# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
```




    GaussianNB(priors=None, var_smoothing=1e-09)




```python
# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)
```


```python
# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
```




    array([[55, 42],
           [12, 91]], dtype=int64)




```python
# Obtener el informe de clasificación
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.82      0.57      0.67        97
               1       0.68      0.88      0.77       103
    
        accuracy                           0.73       200
       macro avg       0.75      0.73      0.72       200
    weighted avg       0.75      0.73      0.72       200
    
    
