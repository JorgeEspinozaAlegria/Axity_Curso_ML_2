![png](../../../imagenes/logotipo-axity-ppt.png)

# Filtrado Basado en Contenido

Los sistemas de recomendación son una colección de algoritmos utilizados para recomendar artículos a los usuarios, basándose en información que proviene del usuario. Estos sistemas se han vuelto omnipresentes y pueden encontrarse en tiendas en línea, dases de datos de películas y buscadores de trabajos.  

En este ejercicio, exploraremos los sistemas de recomendación basados el Filtrado Basado en Contenido.

## Los datos

Utilizaremos el conjunto de datos [Movie Lens de Kaggle](https://www.kaggle.com/shubhammehta21/movie-lens-small-latest-dataset).  

Este contiene las calificaciones de MovieLens, un servicio de recomendación de películas.  

El conjunto contiene 100,000 calificaciones de 1000 usuarios para 1700 películas.

Contiene las siguientes columnas:

**movies.csv**
* movieId: identificador de la película
* title: título de la película
* genres: géneros de la película, separado por |

**ratings.csv**
* userId: Identificador del usuario
* movieId: Identificador de la película
* rating: Calificación (0 - 5)

## Carga de datos y preprocesamiento


```python
# Importemos las librerías
import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

Now let's read each file into their Dataframes:


```python
# Carguemos los datos
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')
```


```python
# Revisemos los datos
movies_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>



Cada pelicula tiene un ID único, un título con el año en que fue liberada y varios géneros en el mismo campo.  
Removamos el año del título y coloquémoslo en una columna adicional.


```python
# Usemos expresiones regulares para encontrar el año almacenado entre paréntesis
# Especificamos los paréntesis, para no tener conflictos con las películas que tienen años en sus títulos
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)

# Removamos los paréntesis
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)

# Removamos el año del título
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')

# Apliquemos la función strip, para quitar los espacios en blanco al final
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
```


```python
# Veamos los resultados
movies_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>Adventure|Children|Fantasy</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men</td>
      <td>Comedy|Romance</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale</td>
      <td>Comedy|Drama|Romance</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II</td>
      <td>Comedy</td>
      <td>1995</td>
    </tr>
  </tbody>
</table>
</div>



Ahora separemos los valores en la columna de géneros en una lista de géneros, para utilizarla posteriormente. Esto puede realizarse mediante la función split de Python.


```python
# Cada género está separado por un | asi que solamente tenemos que llamar la función split con |
movies_df['genres'] = movies_df.genres.str.split('|')
movies_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>[Adventure, Animation, Children, Comedy, Fantasy]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>[Adventure, Children, Fantasy]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men</td>
      <td>[Comedy, Romance]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II</td>
      <td>[Comedy]</td>
      <td>1995</td>
    </tr>
  </tbody>
</table>
</div>



Utilizaremos one hot encoding para separar la lista de géneros, para que los podamos utilizar como características.  
Almacenaremos estos géneros en un dataframe adicional, dado que no lo utilizaremos en nuestro primer sistema de recomendación.  


```python
# Copiemos el dataframe en uno nuevo, dado que no utilizaremos la información para nuestro primer sistema de recomendación
moviesWithGenres_df = movies_df.copy()

# Para cada renglón en el dataframe, iteremos por la lista de géneros y coloquemos un 1 en la columna correspondiente
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1

# Sustituyamos los valores NaN por 0, para indicar que la película no aplica para el género de esa columna
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
moviesWithGenres_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>Fantasy</th>
      <th>Romance</th>
      <th>...</th>
      <th>Horror</th>
      <th>Mystery</th>
      <th>Sci-Fi</th>
      <th>War</th>
      <th>Musical</th>
      <th>Documentary</th>
      <th>IMAX</th>
      <th>Western</th>
      <th>Film-Noir</th>
      <th>(no genres listed)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>[Adventure, Animation, Children, Comedy, Fantasy]</td>
      <td>1995</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>[Adventure, Children, Fantasy]</td>
      <td>1995</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men</td>
      <td>[Comedy, Romance]</td>
      <td>1995</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>1995</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II</td>
      <td>[Comedy]</td>
      <td>1995</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



Ahora revisemos el dataframe de calificaciones


```python
ratings_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>964982703</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>4.0</td>
      <td>964981247</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>6</td>
      <td>4.0</td>
      <td>964982224</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>5.0</td>
      <td>964983815</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>5.0</td>
      <td>964982931</td>
    </tr>
  </tbody>
</table>
</div>



Cada renglón en el dataframe tiene un ID de usuario, asociado con la menos una película, una calificación y una marca de tiempo que indica cuando se calificó. No necesitaremos esta última columna, asi que la eliminaremos.


```python
# Eliminemos la columna de la marca de tiempo
ratings_df = ratings_df.drop('timestamp', 1)
```

Revisemos la versión final del dataframe


```python
ratings_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>6</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



## Filtrado Basado en Contenido

Veamos como implementar el algoritmo **Basado en Contenido (Item-Item)**. Esta técnica trata de identificar los aspectos favoritos de un artículo para un usuario y recomienda artículos que cuentan con esos aspectos. En nuestro caso, trataremos de identificar los géneros favoritos del usuario activo, a partir de las películas y las calificaciones que ha proporcionado.  

Iniciemos con la creación de un usuario activo ficticio, al que le recomendaremos las películas.  

Nota: para agregar mas películas, solo incrementa el número de elementos en la variable *userInput*. Asegúrate de escribir los títulos con la primera letra en mayúscula y si la película inicia con "The", como "The Matrix", escribela de la siguiente manera: 'Matrix, The' .  


```python
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ] 
inputMovies = pd.DataFrame(userInput)
inputMovies
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Breakfast Club, The</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Toy Story</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jumanji</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pulp Fiction</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Akira</td>
      <td>4.5</td>
    </tr>
  </tbody>
</table>
</div>



#### Agregar el ID de la película al usuario activo
Una vez que tenemos completo el usuario activo, extraigamos el ID de la película del dataframe de películas y agreguémoslo.  

Podemos lograr esto, filtrando los renglones que contengan el título de nuestras películas e integrándolos en nuestro dataframe. También eliminaremos las columnas que sean innecesarias.   


```python
# Filtremos las películas por título
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]

# Integrémoslas a nuestro dataframe, para obtener el ID de la película. Se realiza la integración por el título
inputMovies = pd.merge(inputId, inputMovies)

# Eliminemos la información que no utiizaremos
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)

# Dataframe final
# Nota: si agregaste una película al dataframke y no aparece, es probable que no esté en la lista de películas o esté escrita de manera diferente
inputMovies
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>296</td>
      <td>Pulp Fiction</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1274</td>
      <td>Akira</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1968</td>
      <td>Breakfast Club, The</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



Iniciemos con el aprendizaje de las preferencias del usuario. Obtengamos los géneros de las películas que el usuario ha visto, a partir del dataframe que contiene los géneros definidos por valores binarios (es decir, el que obtuvimos al aplicar one hot encoding).


```python
# Incorporemos los géneros a las películas del usuario activo
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
userMovies
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>Fantasy</th>
      <th>Romance</th>
      <th>...</th>
      <th>Horror</th>
      <th>Mystery</th>
      <th>Sci-Fi</th>
      <th>War</th>
      <th>Musical</th>
      <th>Documentary</th>
      <th>IMAX</th>
      <th>Western</th>
      <th>Film-Noir</th>
      <th>(no genres listed)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>[Adventure, Animation, Children, Comedy, Fantasy]</td>
      <td>1995</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>[Adventure, Children, Fantasy]</td>
      <td>1995</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>257</th>
      <td>296</td>
      <td>Pulp Fiction</td>
      <td>[Comedy, Crime, Drama, Thriller]</td>
      <td>1994</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>973</th>
      <td>1274</td>
      <td>Akira</td>
      <td>[Action, Adventure, Animation, Sci-Fi]</td>
      <td>1988</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1445</th>
      <td>1968</td>
      <td>Breakfast Club, The</td>
      <td>[Comedy, Drama]</td>
      <td>1985</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



Solo necesitaremos los géneros separados por columnas, por lo que limpiaremos el dataframe y resetearemos los pindices, eliminando el ID de la película, el título, la columna original de géneros y el año.


```python
# Reseteando los índices
userMovies = userMovies.reset_index(drop=True)

# Eliminando las columnas innecesarias
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
userGenreTable
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>Fantasy</th>
      <th>Romance</th>
      <th>Drama</th>
      <th>Action</th>
      <th>Crime</th>
      <th>Thriller</th>
      <th>Horror</th>
      <th>Mystery</th>
      <th>Sci-Fi</th>
      <th>War</th>
      <th>Musical</th>
      <th>Documentary</th>
      <th>IMAX</th>
      <th>Western</th>
      <th>Film-Noir</th>
      <th>(no genres listed)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Ahora estamos listos para aprender las preferencias del usuario activo.  

Convertiremos cada género en un peso. Podemos realizar esto utilizando las calificaciones que proporcionó el usuario activo y multiplicándolas por la tabla de géneros. Finalmente sumaremos los resultados por columna.  

En términos matemáticos, este es un porducto punto, entre una matriz y un vector, asi que podemos utilizar la función "dot" de pandas.  


```python
# Revisemos las calificaciones del usuario para cada película
inputMovies['rating']
```




    0    3.5
    1    2.0
    2    5.0
    3    4.5
    4    5.0
    Name: rating, dtype: float64




```python
# Realicemos el producto punto
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])

# Revisemos el resultado
userProfile
```




    Adventure             10.0
    Animation              8.0
    Children               5.5
    Comedy                13.5
    Fantasy                5.5
    Romance                0.0
    Drama                 10.0
    Action                 4.5
    Crime                  5.0
    Thriller               5.0
    Horror                 0.0
    Mystery                0.0
    Sci-Fi                 4.5
    War                    0.0
    Musical                0.0
    Documentary            0.0
    IMAX                   0.0
    Western                0.0
    Film-Noir              0.0
    (no genres listed)     0.0
    dtype: float64



Ahora tenemos los pesos por cada preferencia del usuario activo o el perfil del usuario. Mediante este, podemos recomendar las películas que satisfagan las preferencias del usuario.  

Realicemos la extracción de la tabla de géneros del dataframe original.


```python
# Obtengamos los géneros de cada película de nuestro dataframe original
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])

# Eliminemos la información que no es necesaria
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
genreTable.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>Fantasy</th>
      <th>Romance</th>
      <th>Drama</th>
      <th>Action</th>
      <th>Crime</th>
      <th>Thriller</th>
      <th>Horror</th>
      <th>Mystery</th>
      <th>Sci-Fi</th>
      <th>War</th>
      <th>Musical</th>
      <th>Documentary</th>
      <th>IMAX</th>
      <th>Western</th>
      <th>Film-Noir</th>
      <th>(no genres listed)</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
genreTable.shape
```




    (9742, 20)



Con el perfil de usuario y la lista completa de películas y sus géneros, obtengamos el promedio ponderado de cada película, basándonos en el perfil del usuario y recomendemos las primeras 20 películas.  


```python
# Multipliquemos los géneros por los pesos y obtengamos el promedio ponderado
recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
recommendationTable_df.head()
```




    movieId
    1    0.594406
    2    0.293706
    3    0.188811
    4    0.328671
    5    0.188811
    dtype: float64




```python
# Ordenemos las recomendaciones en orden descendente
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)

# Revisemos algunos valores
recommendationTable_df.head()
```




    movieId
    134853    0.734266
    148775    0.685315
    117646    0.678322
    6902      0.678322
    81132     0.671329
    dtype: float64



Con esto ya tenemos nuestra tabla de recomendación


```python
# Generemos la tabla final de recomendaciones
movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>559</th>
      <td>673</td>
      <td>Space Jam</td>
      <td>[Adventure, Animation, Children, Comedy, Fanta...</td>
      <td>1996</td>
    </tr>
    <tr>
      <th>1390</th>
      <td>1907</td>
      <td>Mulan</td>
      <td>[Adventure, Animation, Children, Comedy, Drama...</td>
      <td>1998</td>
    </tr>
    <tr>
      <th>2250</th>
      <td>2987</td>
      <td>Who Framed Roger Rabbit?</td>
      <td>[Adventure, Animation, Children, Comedy, Crime...</td>
      <td>1988</td>
    </tr>
    <tr>
      <th>3460</th>
      <td>4719</td>
      <td>Osmosis Jones</td>
      <td>[Action, Animation, Comedy, Crime, Drama, Roma...</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>4631</th>
      <td>6902</td>
      <td>Interstate 60</td>
      <td>[Adventure, Comedy, Drama, Fantasy, Mystery, S...</td>
      <td>2002</td>
    </tr>
    <tr>
      <th>5490</th>
      <td>26340</td>
      <td>Twelve Tasks of Asterix, The (Les douze travau...</td>
      <td>[Action, Adventure, Animation, Children, Comed...</td>
      <td>1976</td>
    </tr>
    <tr>
      <th>5819</th>
      <td>32031</td>
      <td>Robots</td>
      <td>[Adventure, Animation, Children, Comedy, Fanta...</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>6448</th>
      <td>51939</td>
      <td>TMNT (Teenage Mutant Ninja Turtles)</td>
      <td>[Action, Adventure, Animation, Children, Comed...</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>6455</th>
      <td>52287</td>
      <td>Meet the Robinsons</td>
      <td>[Action, Adventure, Animation, Children, Comed...</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>6462</th>
      <td>52462</td>
      <td>Aqua Teen Hunger Force Colon Movie Film for Th...</td>
      <td>[Action, Adventure, Animation, Comedy, Fantasy...</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>6879</th>
      <td>62956</td>
      <td>Futurama: Bender's Game</td>
      <td>[Action, Adventure, Animation, Comedy, Fantasy...</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>7441</th>
      <td>81132</td>
      <td>Rubber</td>
      <td>[Action, Adventure, Comedy, Crime, Drama, Film...</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>7550</th>
      <td>85261</td>
      <td>Mars Needs Moms</td>
      <td>[Action, Adventure, Animation, Children, Comed...</td>
      <td>2011</td>
    </tr>
    <tr>
      <th>8349</th>
      <td>108540</td>
      <td>Ernest &amp; Célestine (Ernest et Célestine)</td>
      <td>[Adventure, Animation, Children, Comedy, Drama...</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>8357</th>
      <td>108932</td>
      <td>The Lego Movie</td>
      <td>[Action, Adventure, Animation, Children, Comed...</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>8597</th>
      <td>117646</td>
      <td>Dragonheart 2: A New Beginning</td>
      <td>[Action, Adventure, Comedy, Drama, Fantasy, Th...</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>8806</th>
      <td>130520</td>
      <td>Home</td>
      <td>[Adventure, Animation, Children, Comedy, Fanta...</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>8900</th>
      <td>134853</td>
      <td>Inside Out</td>
      <td>[Adventure, Animation, Children, Comedy, Drama...</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>9169</th>
      <td>148775</td>
      <td>Wizards of Waverly Place: The Movie</td>
      <td>[Adventure, Children, Comedy, Drama, Fantasy, ...</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>9282</th>
      <td>157865</td>
      <td>Ratchet &amp; Clank</td>
      <td>[Action, Adventure, Animation, Children, Comed...</td>
      <td>2016</td>
    </tr>
  </tbody>
</table>
</div>



### Ventajas y Desventajas

##### Ventajas
* Aprende las preferencias del usuario
* Altamente personalizado para el usuario

##### Desventajas
* No toma en cuenta lo que otras personas piensan de la película, por lo que se podrían recomendar películas con calificaciones bajas
* La extracción de datos no es siempre intuitiva
* La determinación de que características le gustan o no al usuario no es siempre obvio
