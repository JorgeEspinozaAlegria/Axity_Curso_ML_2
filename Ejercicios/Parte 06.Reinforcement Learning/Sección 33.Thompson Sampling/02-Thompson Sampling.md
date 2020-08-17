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

```


```python
# Visualizar la selección de las máquinas

```


```python

```


![png](../../../imagenes/02-Thompson%20Sampling_10_0.png)



```python
# Visualizar los resultados
print("\nGanancia total = ", total_reward)
```

    
    Ganancia total =  90
    

## Usando Thompson Sampling


```python
# Algoritmo de Muestreo Thompson

```


```python
# Visualizar la selección de las máquinas

```


```python

```


![png](../../../imagenes/02-Thompson%20Sampling_15_0.png)



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


![png](../../../imagenes/02-Thompson%20Sampling_17_0.png)


## ¡Buen trabajo!
