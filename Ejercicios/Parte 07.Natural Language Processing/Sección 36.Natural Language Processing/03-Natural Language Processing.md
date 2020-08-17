![png](../../../imagenes/logotipo-axity-ppt.png)

# Procesamiento de Lenguaje Natural

## Explicación del Ejemplo

El propósito de este ejemplo es construir un modelo, que utilice la última letra de un nombre, para predecir el género.  
En inglés, los nombres que terminan en *a*, *e*, *i*, generalmente son femeninos. Los nombres que terminan en *k*, *o*, *r*, *s*, *t*, generalmente son masculinos.

### Los datos
La librería NLTK en python tiene una lista de nombres en inglés, separadas por género.  

Debido a que contamos con datos etiquetados, podemos utilizar algoritmos de aprendizaje supervisados y obtener la precisión del modelo

## Generar la función para extraer la última letra


```python
def extraer_letra(palabra):
    return {'ultima_letra': palabra[-1]}
```


```python
extraer_letra('John')
```




    {'ultima_letra': 'n'}



## Explorar el corpus de NLTK

NLTK tiene librerías (corpus) con nombres en inglés, separados por género.


```python
# Importemos los nombres
import nltk
nltk.download('names')
from nltk.corpus import names
names.readme().replace('\n', ' ')
```

    'Names Corpus, Version 1.3 (1994-03-29) Copyright (C) 1991 Mark Kantrowitz Additions by Bill Ross  This corpus contains 5001 female names and 2943 male names, sorted alphabetically, one per line.  You may use the lists of names for any purpose, so long as credit is given in any published work. You may also redistribute the list if you provide the recipients with a copy of this README file. The lists are not in the public domain (I retain the copyright on the lists) but are freely redistributable.  If you have any additions to the lists of names, I would appreciate receiving them.  Mark Kantrowitz <mkant+@cs.cmu.edu> http://www-2.cs.cmu.edu/afs/cs/project/ai-repository/ai/areas/nlp/corpora/names/'




```python
# Las librerías (corpora) consisten en conjuntos de archivos, cada uno con una pieza de texto
# la lista de identificadores para estos archivos se puede revisar mediante el método fileids()
# Veamos con que lenguajes contamos
names.fileids()
```




    ['female.txt', 'male.txt']




```python
# Veamos las primeras 10 palabras
print(names.words('female.txt')[:10])
print(names.words('male.txt')[:10])
```

    ['Abagael', 'Abagail', 'Abbe', 'Abbey', 'Abbi', 'Abbie', 'Abby', 'Abigael', 'Abigail', 'Abigale']
    ['Aamir', 'Aaron', 'Abbey', 'Abbie', 'Abbot', 'Abbott', 'Abby', 'Abdel', 'Abdul', 'Abdulkarim']
    

## Construir los conjuntos de datos

Generemos nuestros conjuntos de datos, colocando las etiquetas correspondientes


```python
# Coloquemos las etiquetas según el archivo fuente
nombres_etiquetados = ([(name, 'femenino') for name in names.words('female.txt')] + [(name, 'masculino') for name in names.words('male.txt')])
nombres_etiquetados[:5]
```




    [('Abagael', 'femenino'),
     ('Abagail', 'femenino'),
     ('Abbe', 'femenino'),
     ('Abbey', 'femenino'),
     ('Abbi', 'femenino')]




```python
# Desordenemos los datos, para que se puedan generar conjuntos de datos de entrenamiento y pruebas
import random
random.shuffle(nombres_etiquetados)
nombres_etiquetados[:5]
```




    [('Valene', 'femenino'),
     ('Gladi', 'femenino'),
     ('Barbi', 'femenino'),
     ('Bev', 'femenino'),
     ('Annetta', 'femenino')]




```python
# Generemos la lista con la última letra
caracteristicas = [(extraer_letra(n), genero) for (n, genero) in nombres_etiquetados]
caracteristicas[:5]
```




    [({'ultima_letra': 'e'}, 'femenino'),
     ({'ultima_letra': 'i'}, 'femenino'),
     ({'ultima_letra': 'i'}, 'femenino'),
     ({'ultima_letra': 'v'}, 'femenino'),
     ({'ultima_letra': 'a'}, 'femenino')]




```python
# Veamos cuantos datos tiene nuestro conjunto de datos
len(caracteristicas)
```




    7944



## Entrenar el modelo


```python
# Importemos las librerias
from nltk import NaiveBayesClassifier
```


```python
# Dividamos el conjunto de datos en conjunto de entrenamiento y conjunto de prueba
from nltk.classify import apply_features
porcentaje = round(len(caracteristicas) * .8)
X_train = apply_features(extraer_letra, nombres_etiquetados[porcentaje:])
X_test = apply_features(extraer_letra, nombres_etiquetados[:porcentaje])
```


```python
# Ajustar el clasificador en el Conjunto de Entrenamiento
classifier = NaiveBayesClassifier.train(X_train)
```


```python
# Veamos la precisión del modelo
print(nltk.classify.accuracy(classifier, X_test))
```

    0.7612903225806451
    


```python
# Veamos algunas predicciones
classifier.classify(extraer_letra('Trinity'))
```




    'femenino'




```python
classifier.classify(extraer_letra('Neo'))
```




    'masculino'


