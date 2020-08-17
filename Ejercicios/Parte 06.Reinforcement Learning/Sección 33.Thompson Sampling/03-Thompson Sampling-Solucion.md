![png](../../../imagenes/logotipo-axity-ppt.png)

# Thompson Sampling

## Planteamiento del Problema

Supongamos que llegamos a un casino, en el que tenemos varias máquinas tragamonedas. El objetivo es maximizar la ganancia que una persona podría obtener, si juega de manera aleatoria en las máquinas.  

Utilizaremos el aprendizaje por reforzamiento, mediante el método Thompson Sampling, para encontrar la máquina que proporcionará las mayores ganancias.  

### Los datos
Generaremos un conjunto de datos aleatorio, con 200 observaciones de 5 máquinas tragamonedas.  

El conjunto de datos contiene la recompensa de cada máquina en cada observación. Es decir, si el jugador ganó, la recompensa es 1; si el jugador perdió, la recompensa es 0.  

Basándonos en estas recompensas, la tarea es identificar cuál de las máquinas proporcionará mayores ganancias.

## Generación de Datos


```python
# Importar las librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
# Generar los datos
import random
data = {}
data['B1'] = [random.randint(0,1) for x in range(200)]
data['B2'] = [random.randint(0,1) for x in range(200)]
data['B3'] = [random.randint(0,1) for x in range(200)]
data['B4'] = [random.randint(0,1) for x in range(200)]
data['B5'] = [random.randint(0,1) for x in range(200)]
dataset = pd.DataFrame(data)
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
      <th>B1</th>
      <th>B2</th>
      <th>B3</th>
      <th>B4</th>
      <th>B5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Selección Aleatoria

Para validar la eficiencia del algoritmo a utilizar, generemos una selección aleatoria de las máquinas.  


```python
# Implementar una Selección Aleatoria
import random
N = 200
d = 5
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward
```


```python
# Visualizar la selección de las máquinas
plt.hist(ads_selected)
plt.xlabel("Máquina")
plt.ylabel("Frecuencia de selección de la máquina")
plt.show()
```


![png](../../../imagenes/03-Thompson%20Sampling-Solucion_8_0.png)



```python
# Visualizar los resultados
print("\nGanancia total = ", total_reward)
```

    
    Ganancia total =  90
    

## Usando Thompson Sampling


```python
# Algoritmo de Muestreo Thompson
import random
N = 200
d = 5
number_of_rewards_1 = [0] * d
number_of_rewards_0 = [0] * d
ads_selected = []
total_reward = 0
for n in range(0, N):
    max_random = 0
    ad = 0
    for i in range(0, d):
        random_beta = random.betavariate(number_of_rewards_1[i]+1, number_of_rewards_0[i]+1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
    else:
        number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1
    total_reward = total_reward + reward
```


```python
# Visualizar la selección de las máquinas
plt.hist(ads_selected)
plt.xlabel("Máquina")
plt.ylabel("Frecuencia de selección de la máquina")
plt.show()
```


![png](../../../imagenes/03-Thompson%20Sampling-Solucion_12_0.png)



```python
# Visualizar los resultados
print("\n\nGanancia por máquina = ", number_of_rewards_1)
print("\nGanancia total = ", total_reward)
```

    
    
    Ganancia por máquina =  [21, 1, 35, 30, 6]
    
    Ganancia total =  93
    


```python
# Visualizar las ganancias por máquina
plt.bar(['B1','B2','B3','B4','B5'],number_of_rewards_1)
plt.xlabel('Máquina')
plt.ylabel('Ganancia por Máquina')
plt.show()
```


![png](../../../imagenes/03-Thompson%20Sampling-Solucion_14_0.png)


## ¡Buen trabajo!
