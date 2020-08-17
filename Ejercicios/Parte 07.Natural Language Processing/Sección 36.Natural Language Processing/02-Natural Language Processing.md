![png](../../../imagenes/logotipo-axity-ppt.png)

# Procesamiento de Lenguaje Natural

## Explicación del Ejemplo

El propósito de este ejemplo es construir un modelo, que detecte el lenguaje en el que está escrito un texto, basándose en las **stop words**.  

El pre-procesamiento es el proceso de convertir datos en algo que pueda entender una computadora. Una de las formas de preprocesamiento más utilizadas, es el filtrado de datos que no son útiles. En PLN, las palabras no útiles son llamdas **stop words**.  
Las **stop words** son palabras comunmente utilizadas (como "el", "la", "un", "una", "en"), que generalmente no aportan valor para el proceso. Por ejemplo en máquinas de búsqueda, se ignoran para que no sean consideradas en los resultados que arroje una consulta.  

La librería NLTK en python tiene una lista de estas palabras en 16 lenguajes diferentes.  

### Los datos
Al inicio proporcionaremos una frase en alguno de los 16 lenguajes de la librería.


**Nota: Este ejercicio no utiliza ningún algoritmo de ML. El propósito es mostrar técnicas de pre-procesamiento y aprovechar las librerías de nltk**  
Asegúrate de contar con la librería nltk antes de iniciar

## Generar la frase y sus tokens


```python
# Generar la frase a probar
texto = "La primer obligación de todo ser humano es ser feliz, la segunda es hacer feliz a los demás"
```


```python
# Importar las librerias
import sys

# librería que separa los textos, basándo en espacios y signos de puntuación (excepto guión bajo)
from nltk.tokenize import wordpunct_tokenize
```


```python
tokens = wordpunct_tokenize (texto)
tokens
```




    ['La',
     'primer',
     'obligación',
     'de',
     'todo',
     'ser',
     'humano',
     'es',
     'ser',
     'feliz',
     ',',
     'la',
     'segunda',
     'es',
     'hacer',
     'feliz',
     'a',
     'los',
     'demás']



Existen otros métodos para generar tokens, por ejemplo, RegexpTokenizer (en el que se puede indicar el separador), WhitespaceTokenizer (similar al string.split() de python) y BlanklineTokenizer.

## Explorar el corpus de NLTK

NLTK tiene librerías (corpus) con **stop words** en varios lenguajes.


```python
# Importemos las stop words
from nltk.corpus import stopwords
```


```python
# Las librerías (corpora) consisten en conjuntos de archivos, cada uno con una pieza de texto
# la lista de identificadores para estos archivos se puede revisar mediante el método fileids()
# Veamos con que lenguajes contamos
stopwords.fileids()
```




    ['arabic',
     'azerbaijani',
     'danish',
     'dutch',
     'english',
     'finnish',
     'french',
     'german',
     'greek',
     'hungarian',
     'indonesian',
     'italian',
     'kazakh',
     'nepali',
     'norwegian',
     'portuguese',
     'romanian',
     'russian',
     'slovene',
     'spanish',
     'swedish',
     'tajik',
     'turkish']




```python
# Contamos con varios métodos para explorar las librerías
# Veamos el contenido del archivo de un lenguaje
stopwords.raw('russian')
```




    'и\nв\nво\nне\nчто\nон\nна\nя\nс\nсо\nкак\nа\nто\nвсе\nона\nтак\nего\nно\nда\nты\nк\nу\nже\nвы\nза\nбы\nпо\nтолько\nее\nмне\nбыло\nвот\nот\nменя\nеще\nнет\nо\nиз\nему\nтеперь\nкогда\nдаже\nну\nвдруг\nли\nесли\nуже\nили\nни\nбыть\nбыл\nнего\nдо\nвас\nнибудь\nопять\nуж\nвам\nведь\nтам\nпотом\nсебя\nничего\nей\nможет\nони\nтут\nгде\nесть\nнадо\nней\nдля\nмы\nтебя\nих\nчем\nбыла\nсам\nчтоб\nбез\nбудто\nчего\nраз\nтоже\nсебе\nпод\nбудет\nж\nтогда\nкто\nэтот\nтого\nпотому\nэтого\nкакой\nсовсем\nним\nздесь\nэтом\nодин\nпочти\nмой\nтем\nчтобы\nнее\nсейчас\nбыли\nкуда\nзачем\nвсех\nникогда\nможно\nпри\nнаконец\nдва\nоб\nдругой\nхоть\nпосле\nнад\nбольше\nтот\nчерез\nэти\nнас\nпро\nвсего\nних\nкакая\nмного\nразве\nтри\nэту\nмоя\nвпрочем\nхорошо\nсвою\nэтой\nперед\nиногда\nлучше\nчуть\nтом\nнельзя\nтакой\nим\nболее\nвсегда\nконечно\nвсю\nмежду\n'




```python
# Sustituyamos los saltos de línea por espacios, para facilitar la visualización
stopwords.raw('russian').replace('\n',' ')
```




    'и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между '




```python
# Veamos las primeras 10 palabras
stopwords.words('spanish')[:10]
```




    ['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se']




```python
# Veamos cuantas palabras tenemos en un lenguaje
print(len(stopwords.words(['spanish'])))
print(len(stopwords.words(['spanish', 'english'])))
```

    313
    492
    

Hay un total de 313 palabras en español y 492 en español e inglés

## Realizar la clasificación

Realizaremos un ciclo a través de las **stop words** en todos los lenguajes y verificaremos cuántas están presentes en el texto de prueba.  
Después determinaremos el lenguaje de nuestro texto, basándonos en el lenguaje que contenga más **stop words**.


```python
lenguaje_conteo = {}

palabras = [word.lower() for word in tokens] # comvertir a minúsculas
palabras_conjunto = set(palabras)

for lenguaje in stopwords.fileids():
    stopwords_conjunto = set(stopwords.words(lenguaje))
    elementos_comunes = palabras_conjunto.intersection(stopwords_conjunto)
    lenguaje_conteo[lenguaje] = len(elementos_comunes)
    
lenguaje_conteo
```




    {'arabic': 0,
     'azerbaijani': 1,
     'danish': 1,
     'dutch': 1,
     'english': 1,
     'finnish': 0,
     'french': 3,
     'german': 1,
     'greek': 0,
     'hungarian': 2,
     'indonesian': 0,
     'italian': 2,
     'kazakh': 0,
     'nepali': 0,
     'norwegian': 1,
     'portuguese': 2,
     'romanian': 3,
     'russian': 0,
     'slovene': 0,
     'spanish': 6,
     'swedish': 1,
     'tajik': 0,
     'turkish': 1}




```python
# Dentro de la función max(), el parámetro key es una función que realiza un cómputo con la llave.
# En este caso, como ya tenemos una llave, asignanos el parámetro key como lenguaje_conteo.get, que regresa una llave
lenguaje_identificado = max(lenguaje_conteo, key=lenguaje_conteo.get)
lenguaje_identificado
```




    'spanish'




```python
# Veamos que stopwords se encontraron
palabras_conjunto.intersection(set(stopwords.words(lenguaje_identificado)))
```




    {'a', 'de', 'es', 'la', 'los', 'todo'}


