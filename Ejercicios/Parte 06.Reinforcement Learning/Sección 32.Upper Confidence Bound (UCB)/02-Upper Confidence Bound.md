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

```


```python

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

```


```python
# Visualizar la selección de las máquinas

```


```python

```


![png](../../../imagenes/02-Upper%20Confidence%20Bound_10_0.png)



```python
# Visualizar los resultados
print("\nGanancia total = ", total_reward)
```

    
    Ganancia total =  101
    

## Usando Upper Confidence Bound (UCB)


```python
# Algoritmo de UCB

```


```python
# Visualizar la selección de las máquinas

```


```python

```


![png](../../../imagenes/02-Upper%20Confidence%20Bound_15_0.png)



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


![png](../../../imagenes/02-Upper%20Confidence%20Bound_17_0.png)


## ¡Buen trabajo!
