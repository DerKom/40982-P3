# Práctica 3. Detección y reconocimiento de formas

Este repositorio contiene la **Práctica 3** donde se utilizan técnicas de procesamiento de imágenes con **OpenCV** para la detección y caracterización geométrica de objetos. 

# Índice

- [Práctica 3. Detección y reconocimiento de formas](#práctica-3-detección-y-reconocimiento-de-formas)
- [Índice](#índice)
  - [Librerías utilizadas](#librerías-utilizadas)
  - [Autores](#autores)
  - [Tareas](#tareas)
    - [Tarea 1](#tarea-1)
      - [Definición de Diámetros de Monedas](#definición-de-diámetros-de-monedas)
      - [Redimensionar la Imagen](#redimensionar-la-imagen)
      - [Conversión a Escala de Grises y Suavizado](#conversión-a-escala-de-grises-y-suavizado)
      - [Detección de Círculos](#detección-de-círculos)
      - [Almacenar Monedas Detectadas](#almacenar-monedas-detectadas)
      - [Selección de Moneda de Referencia con Click](#selección-de-moneda-de-referencia-con-click)
      - [Identificación de Monedas Restantes](#identificación-de-monedas-restantes)
      - [Mostrar Resultado y Calcular Suma Total](#mostrar-resultado-y-calcular-suma-total)
      - [Resultados](#resultados)
    - [Tarea 2 - Clasificación de Partículas](#tarea-2---clasificación-de-partículas)
      - [Introducción](#introducción)
      - [Conjunto de Datos](#conjunto-de-datos)
      - [Metodología](#metodología)
        - [1. Preprocesamiento de Imágenes](#1-preprocesamiento-de-imágenes)
        - [2. Segmentación](#2-segmentación)
        - [3. Detección de Contornos](#3-detección-de-contornos)
        - [4. Extracción de Características](#4-extracción-de-características)
        - [5. Clasificación](#5-clasificación)
        - [6. Evaluación](#6-evaluación)
      - [Resultados](#resultados-1)
      - [Matriz de Confusión](#matriz-de-confusión)
  - [Referencias y bibliografía](#referencias-y-bibliografía)

## Librerías utilizadas

[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy)](https://numpy.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=matplotlib)](https://matplotlib.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=seaborn)](https://seaborn.pydata.org/)


## Autores

Este proyecto fue desarrollado por:

- [![GitHub](https://img.shields.io/badge/GitHub-Francisco%20Javier%20L%C3%B3pez%E2%80%93Dufour%20Morales-green?style=flat-square&logo=github)](https://github.com/gitfrandu4)
- [![GitHub](https://img.shields.io/badge/GitHub-Marcos%20V%C3%A1zquez%20Tasc%C3%B3n-blue?style=flat-square&logo=github)](https://github.com/DerKom)

## Tareas

### Tarea 1
La tarea consiste en localizar las monedas dentro de una imagen y realizar la suma de sus valores. Para ello, seleccionamos la moneda de 1eur como referencia y el programa logra la funcionalidad deseada de la siguiente manera:

#### Definición de Diámetros de Monedas
```python
diametros_monedas = {
    '1 cent': 16.25,
    '2 cent': 18.75,
    '5 cent': 21.25,
    '10 cent': 19.75,
    '20 cent': 22.25,
    '50 cent': 24.25,
    '1 eur': 23.25,
    '2 eur': 25.75
}
```
Comenzamos con la definición de un diccionario que almacena los diámetros reales de diferentes monedas en milímetros.

#### Redimensionar la Imagen
```python
scale_width = screen_width / width
scale_height = screen_height / height
scale = min(scale_width, scale_height)
imagen_redimensionada = cv2.resize(imagen, (new_width, new_height), interpolation=cv2.INTER_AREA)
```
El tamaño de la imagen se ajusta para que quepa en una ventana con dimensiones predeterminadas (1280x720), manteniendo la relación de aspecto.

#### Conversión a Escala de Grises y Suavizado
```python
imgGris = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2GRAY)
imgGris_suavizada = cv2.GaussianBlur(imgGris, (11, 11), 2)
```
Se convierte la imagen a escala de grises para simplificar la detección de bordes. Luego, se aplica un suavizado Gaussian Blur para reducir el ruido en la imagen.

#### Detección de Círculos
```python
monedasDetectadas = cv2.HoughCircles(
    imgGris_suavizada,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=100,
    param1=100,
    param2=60,
    minRadius=40,
    maxRadius=150
)
```
Se utiliza el método `HoughCircles` para detectar los círculos (monedas) en la imagen. Esta función busca patrones circulares y detecta sus posiciones y radios.

#### Almacenar Monedas Detectadas
```python
coins = []
for i in monedasDetectadas[0, :]:
    coin = {
        'center': (i[0], i[1]),
        'radius': i[2],
        'value': None,
        'diameter_px': i[2] * 2
    }
    coins.append(coin)
```
Los datos de las monedas detectadas se almacenan en una lista de diccionarios, donde se registra el centro, radio y diámetro en píxeles de cada moneda.

#### Selección de Moneda de Referencia con Click
```python
def onclick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for coin in coins:
            dx = x - coin['center'][0]
            dy = y - coin['center'][1]
            distancia_cuadrada = dx ** 2 + dy ** 2
            if distancia_cuadrada <= coin['radius'] ** 2:
                print("Moneda seleccionada como referencia (1 EUR)")
                pixel_mm_ratio = coin['diameter_px'] / diametros_monedas['1 eur']
                identificar_monedas()
                break
```
Se define una función que detecta clics del usuario en la imagen. Cuando el usuario hace clic sobre una moneda, esta se toma como referencia (moneda de 1 EUR) y se calcula la relación píxeles/mm.

#### Identificación de Monedas Restantes
```python
def identificar_monedas():
    for coin in coins:
        if coin['value'] is None:
            diametro_mm = coin['diameter_px'] / pixel_mm_ratio
            idx = (np.abs(diametros_reales - diametro_mm)).argmin()
            coin['value'] = valores_monedas[idx]
            print(f"Moneda identificada: {coin['value']}")
```
Una vez que se ha seleccionado una moneda de referencia, el programa calcula el tamaño de las otras monedas y las compara con los diámetros conocidos para identificar su valor.

#### Mostrar Resultado y Calcular Suma Total
```python
total_sum = 0
for coin in coins:
    if coin['value'] is not None:
        value_str = coin['value']
        if 'cent' in value_str:
            value = int(value_str.split(' ')[0]) / 100
        elif 'eur' in value_str:
            value = float(value_str.split(' ')[0])
        total_sum += value
```
Se calcula y muestra la suma total del valor de todas las monedas detectadas.

#### Resultados
<table>
  <tr>
    <td style="text-align: center; padding-right: 20px;">
      <img src="results/task1-result1.png" width="200" alt="Resultado Tarea 1.1">
    </td>
    <td style="text-align: center;">
      <img src="results/task1-result2.png" width="450" alt="Resultado Tarea 1.2">
    </td>
  </tr>
</table>

### Tarea 2 - Clasificación de Partículas

#### Introducción

En esta tarea abordamos la clasificación de partículas encontradas en playas de Canarias, centrándonos en tres categorías: **fragmentos plásticos**, **pellets** y **alquitrán**. Utilizamos imágenes de cada categoría para extraer características geométricas y de apariencia que faciliten su clasificación.

Puede ejecutar el cuaderno de Jupyter `VC_P3_T2.ipynb` para ver el código y los resultados detallados.

#### Conjunto de Datos

Trabajamos con tres imágenes, cada una correspondiente a una categoría específica:

- **Fragmentos**: Plásticos de menos de 5 mm (microplásticos).
- **Pellets**: Partículas esféricas plásticas de menos de 5 mm.
- **Alquitrán**: Restos de petróleo comunes en las playas.

![Imagen de Fragmentos](results/fragmentos.png)

#### Metodología

##### 1. Preprocesamiento de Imágenes

- **Conversión a Escala de Grises**: Simplificamos las imágenes para el procesamiento.
- **Desenfoque Gaussiano**: Reducimos el ruido y mejoramos la segmentación.
- **Apertura Morfológica**: Eliminamos pequeños objetos y ruido residual.

##### 2. Segmentación

- **Umbralización de Otsu**: Segmentamos las partículas del fondo.
- **Inversión de Imágenes**: Aseguramos que las partículas sean blancas sobre fondo negro.

##### 3. Detección de Contornos

- **Encontrar Contornos**: Identificamos los contornos de las partículas.
- **Filtrado por Área**: Eliminamos contornos que no corresponden a partículas reales estableciendo umbrales de área.

##### 4. Extracción de Características

Para cada partícula detectada, extraemos:

- Área
- Perímetro
- Compacidad (`C = P² / A`)
- Relación de Aspecto (ancho/alto)
- Extensión (área de la partícula / área del contenedor)
- Relación de Ejes de la Elipse Ajustada

##### 5. Clasificación

- **Clasificador Basado en Reglas**: Implementamos un clasificador manual utilizando umbrales derivados del análisis estadístico.
- **Asignación de Etiquetas**: Clasificamos cada partícula en una de las tres categorías.

##### 6. Evaluación

- **Matriz de Confusión**: Evaluamos el rendimiento del clasificador.
- **Informe de Clasificación**: Calculamos métricas como precisión, recall y F1-score.

#### Resultados

#### Matriz de Confusión

<img src='results/confusion_matrix.png' width='400'>
<!-- <img src='results/confusion_matrix_2.png' width='400'> -->


## Referencias y bibliografía

- OpenCV Documentation: https://docs.opencv.org/
- NumPy Documentation: https://numpy.org/doc/
- Matplotlib Documentation: https://matplotlib.org/stable/contents.html
- OpenCV Documentation: [docs.opencv.org](https://docs.opencv.org/)
- NumPy Documentation: [numpy.org/doc](https://numpy.org/doc/)
