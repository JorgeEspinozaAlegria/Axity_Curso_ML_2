![png](../../../imagenes/logotipo-axity-ppt.png)

# Upper Confidence Bound (UCB) - Solución

## Planteamiento del Problema

Supongamos que llegamos a un casino, en el que tenemos varias máquinas tragamonedas. El objetivo es maximizar la ganancia que una persona podría obtener, si juega de manera aleatoria en las máquinas.  

Utilizaremos el aprendizaje por reforzamiento, mediante el método UCB, para encontrar la máquina que proporcionará las mayores ganancias.  

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
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
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


![png](../../../imagenes/03-Upper%20Confidence%20Bound-Solucion_8_0.png)



```python
# Visualizar los resultados
print("\nGanancia total = ", total_reward)
```

    
    Ganancia total =  101
    

## Usando Upper Confidence Bound (UCB)


```python
# Algoritmo de UCB
import math
N = 200
d = 5
number_of_selections = [0] * d
sums_of_rewards = [0] * d
ads_selected = []
total_reward = 0
for n in range(0, N):
    max_upper_bound = 0
    ad = 0
    for i in range(0, d):
        if(number_of_selections[i]>0):
            average_reward = sums_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt(3/2*math.log(n+1)/number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
            
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
```


```python
# Visualizar la selección de las máquinas
plt.hist(ads_selected)
plt.xlabel("Máquina")
plt.ylabel("Frecuencia de selección de la máquina")
plt.show()
```


![png](../../../imagenes/03-Upper%20Confidence%20Bound-Solucion_12_0.png)



```python
# Visualizar los resultados
print("\n\nGanancia por máquina = ", sums_of_rewards)
print("\nGanancia total = ", total_reward)
```

    
    
    Ganancia por máquina =  [13, 25, 7, 21, 37]
    
    Ganancia total =  103
    


```python
# Visualizar las ganancias por máquina
plt.bar(['B1','B2','B3','B4','B5'],sums_of_rewards)
plt.xlabel('Máquina')
plt.ylabel('Ganancia por Máquina')
plt.show()
```


![png](../../../imagenes/03-Upper%20Confidence%20Bound-Solucion_14_0.png)


## ¡Buen trabajo!
