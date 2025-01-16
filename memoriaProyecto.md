# Memoria Completa del Proyecto: Clasificación de Sonidos con CNN utilizando el Dataset ESC-50

## Introducción
El proyecto buscó abordar un desafío técnico relacionado con la clasificación de sonidos utilizando redes neuronales convolucionales (CNN) y el dataset **ESC-50**, el cual consta de 2,000 clips de audio categorizados en 50 clases. Este documento tiene como objetivo registrar de forma exhaustiva el progreso, las decisiones y las estrategias adoptadas a lo largo del desarrollo.

El trabajo se enfocó en:

- Diseñar un modelo propio desde cero y evaluar su desempeño.
- Explorar modelos preentrenados mediante Transfer Learning.
- Solucionar los retos inherentes a trabajar con un dataset auditivo reducido y balanceado.
- Mejorar continuamente la calidad de los resultados a través de ajustes en el preprocesamiento y la arquitectura del modelo.

## Definición del Problema y Selección del Dataset

### Justificación
Durante la etapa inicial del proyecto, se evaluaron diversas opciones de datasets y enfoques potenciales:

1. **Análisis de emociones en texto:** Modelos de clasificación de sentimientos en mensajes o reseñas.
2. **Predicción de series temporales:** Identificación de patrones en datos climáticos o financieros.
3. **Diagnóstico médico:** Clasificación de imágenes médicas para detección de anomalías.

El dataset **ESC-50** fue seleccionado por su complejidad y desafíos inherentes:

- **Naturaleza auditiva:** Requiere trabajar con datos de audio procesados en representaciones visuales (espectrogramas).
- **Distribución balanceada:** Cada clase cuenta con 40 ejemplos, facilitando comparaciones entre categorías.
- **Clases diversas:** Incluye sonidos naturales, animales y humanos, lo que demanda una buena capacidad de generalización.

A pesar de estas ventajas, también presenta retos significativos:

- **Tamaño limitado:** La cantidad de datos puede resultar insuficiente para modelos complejos sin augmentación.
- **Similitud intraclase:** Ejemplos como "lluvia" y "olas" comparten patrones visuales y auditivos similares.

---

## Desarrollo del Proyecto

### Preprocesamiento de Datos

#### Metodología Inicial
El preprocesamiento fue un paso crítico para transformar los clips de audio en un formato compatible con las redes neuronales. Las principales tareas incluyeron:

1. **Estandarización de la frecuencia de muestreo:** Todos los clips se procesaron a 22,050 Hz para garantizar la uniformidad.
2. **Generación de espectrogramas de Mel:** Utilizando la librería `librosa`, se crearon representaciones visuales de los clips con 128 bandas.
3. **Normalización adecuada:** Se eliminó la doble normalización y se empleó la función `power_to_db` para escalar los valores.
4. **Dimensiones consistentes:** Los espectrogramas se ajustaron a un tamaño fijo de 128x215 mediante padding o truncamiento, dependiendo de la duración del clip.
5. **Manejo robusto de errores:** Se implementaron bloques `try-except` para gestionar clips corruptos o con problemas en la carga sin interrumpir el flujo general del preprocesamiento.

#### Problemas y Soluciones

1. **Variaciones en la duración:** Los clips de duración inconsistente se trataron mediante padding con valores constantes o truncamiento.
2. **Fallos en la carga de archivos:** Archivos corruptos o con formato incompatible se ignoraron mediante manejo de excepciones.
3. **Resolución visual insuficiente:** Se optimizaron parámetros como `n_fft=2048` y `hop_length=512` para mejorar la calidad visual de los espectrogramas.

---

### Diseño del Modelo Propio

#### Arquitectura Inicial
El modelo se diseñó con los siguientes componentes:

1. **Capas convolucionales:** Tres bloques de convoluciones con filtros incrementales (32, 64, 128), cada uno seguido de BatchNormalization y MaxPooling.
2. **Capas densas:** Dos capas completamente conectadas con Dropout (0.3) y regularización L2 (0.01).
3. **Salida:** Una capa softmax con 50 neuronas para clasificar en las 50 clases del dataset.
4. **Optimizador:** Se utilizó Adam con la función de pérdida `categorical_crossentropy`.

#### Desafíos Identificados

1. **Sobreajuste:** El modelo mostraba una diferencia significativa entre el rendimiento en entrenamiento y validación.
   - **Solución:** Se redujeron los valores de Dropout y L2 a 0.2 y 0.001, respectivamente, para minimizar el exceso de regularización.

2. **Configuración del kernel:** El uso de un kernel grande (7x7) en la primera capa resultó en pérdida de detalles críticos.
   - **Solución:** Se ajustó el kernel a 3x3 con un `stride` de 1 para preservar la información.

3. **Estancamiento en etapas tempranas:** Las primeras épocas mostraban convergencia lenta y baja mejora en la validación.
   - **Solución:** Se introdujeron estrategias como Early Stopping y ReduceLROnPlateau para optimizar el entrenamiento.

---

### Exploración de Transfer Learning

#### Evaluación de Arquitecturas Preentrenadas
Se probaron varias arquitecturas reconocidas:

1. **ResNet50:** Utilizando pesos preentrenados en ImageNet, se congelaron las capas base y se añadió un cabezal personalizado. Sin embargo, la adaptación a espectrogramas no fue satisfactoria.
2. **InceptionV3:** Mostró un desempeño ligeramente mejor que ResNet50, pero aún inferior al modelo propio.
3. **EfficientNetB3:** Esta arquitectura demostró una capacidad superior para ajustar los patrones del dataset tras realizar un fine-tuning en las últimas capas.

#### Retos en Transfer Learning

- **Especialización:** Modelos como ResNet50 e InceptionV3, diseñados para datos visuales generales, presentaron dificultades para manejar espectrogramas.
- **Adaptación:** Se identificó la necesidad de arquitecturas optimizadas para audio, como **VGGish**, **Conformer**, y **AST (Audio Spectrogram Transformer)**.

---

### Herramientas de Evaluación y Visualización

Se desarrollaron recursos adicionales para entender y mejorar el rendimiento:

1. **Matriz de confusión:** Permitía identificar clases problemáticas, como la confusión entre "lluvia" y "olas".
2. **Visualización de espectrogramas:** Se implementaron funciones para inspeccionar visualmente los casos mal clasificados.
3. **Reportes detallados:** Generación de métricas como precisión y recall para evaluar el comportamiento por clase.

---

### Implementaciones Avanzadas

#### Optimización del Pipeline de Entrenamiento
A medida que el proyecto avanzaba, se realizaron mejoras para refinar el flujo de entrenamiento y reducir problemas encontrados durante las primeras iteraciones. Esto incluyó ajustes en los hiperparámetros, integración de nuevas técnicas de optimización, y la incorporación de data augmentation en el pipeline de entrenamiento.

1. **Ajuste de Hiperparámetros:**
   - Se implementó un programador de tasa de aprendizaje (`LearningRateScheduler`) que permitía ajustar dinámicamente el learning rate a lo largo de las épocas. Esto mejoró significativamente la estabilidad durante el entrenamiento y evitó saltos en los mínimos locales.
   - Se experimentó con varios valores de `batch_size`, determinando que un tamaño de 32 ofrecía el mejor equilibrio entre rendimiento computacional y convergencia del modelo.

2. **Data Augmentation en Línea:**
   - Se configuró un pipeline utilizando `tf.data` que realizaba transformaciones en tiempo real, como rotaciones, zoom, y traslaciones sobre los espectrogramas. Esto incrementó la diversidad de datos para el entrenamiento, mitigando los efectos del sobreajuste.

3. **Regularización y Simplificación del Modelo:**
   - Se redujo la complejidad de las capas densas finales para minimizar el riesgo de sobreajuste. La arquitectura final incluyó:
     - Dropout ajustado a 0.2 para las capas densas.
     - Regularización L2 en todas las capas convolucionales y densas, con un valor de 0.001.

#### Exploración de Modelos Específicos para Audio
Dado el rendimiento limitado de modelos generales en las primeras pruebas, se exploraron alternativas especializadas para datos auditivos. Entre ellas:

1. **VGGish:** Arquitectura preentrenada en AudioSet, diseñada para la representación de espectrogramas de audio. Su rendimiento inicial fue prometedor, aunque con limitaciones en ciertas clases.

2. **Conformer:** Modelo que combina convoluciones y transformers, mostrando una capacidad sobresaliente para capturar relaciones temporales y frecuenciales en los datos de audio.

3. **AST (Audio Spectrogram Transformer):** Diseñado específicamente para espectrogramas, este modelo demostró ser una de las opciones más prometedoras, logrando adaptarse bien a las peculiaridades del dataset ESC-50.

#### Fine-Tuning y Comparación de Modelos
Todos los modelos preentrenados se sometieron a un proceso de fine-tuning. Este incluyó:

- Congelar capas iniciales y ajustar únicamente las capas finales.
- Modificar los optimizadores y sus hiperparámetros según el modelo. Por ejemplo, **AST** mostró mejor desempeño con optimizadores basados en learning rate adaptativo.
- Evaluar cada modelo en las mismas condiciones utilizando k-fold cross-validation para obtener métricas más robustas.

---

### Proceso Iterativo de Evaluación

#### Identificación de Errores Comunes
Se implementaron métricas detalladas y herramientas visuales para analizar los errores:

1. **Matriz de Confusión Extendida:**
   - Proporcionó información clave sobre las clases más problemáticas.
   - Por ejemplo, se observó una confusión recurrente entre las clases "olas" y "lluvia", debido a la similitud en sus espectrogramas.

2. **Análisis Visual de Casos Mal Clasificados:**
   - Espectrogramas mal clasificados se inspeccionaron junto con sus predicciones y etiquetas verdaderas, lo que permitió identificar patrones de error específicos.

3. **Errores de Sesgo en las Predicciones:**
   - Se analizó si el modelo favorecía ciertas clases o si algunas estaban subrepresentadas en las predicciones finales.

#### Ajustes Basados en Resultados
Con base en los análisis realizados:

- Se ajustaron los parámetros de generación de espectrogramas, priorizando características que destacaran diferencias intraclase.
- Se probó la inclusión de augmentaciones específicas, como agregar ruido blanco o modificar la velocidad del audio, para diversificar los datos de entrenamiento.

---

### Implementación de Propuestas Finales

#### Data Augmentation Específico
En la etapa final del proyecto, se integraron técnicas de data augmentation enfocadas en características específicas del dominio de audio:

1. **Modificación del Espectro de Frecuencias:**
   - Se implementaron técnicas para ajustar el pitch y la velocidad del audio de forma controlada, simulando escenarios de grabación reales donde estas variables podrían variar naturalmente.
   - Ejemplo de código:
     ```python
     def time_stretch(audio, rate=0.8):
         return librosa.effects.time_stretch(audio, rate=rate)
     ```

2. **Inyección de Ruido:**
   - Se añadió ruido blanco controlado para robustecer el modelo frente a grabaciones con condiciones de fondo diversas.

3. **Masking Aleatorio en Espectrogramas:**
   - Inspirado en técnicas de visión por computadora, se enmascararon partes del espectrograma para simular datos incompletos.
   - Ejemplo:
     ```python
     def apply_mask(spec, mask_size):
         mask_start = np.random.randint(0, spec.shape[1] - mask_size)
         spec[:, mask_start:mask_start+mask_size] = 0
         return spec
     ```

#### Refinamiento del Modelo
Tras integrar estas transformaciones, se realizaron ajustes adicionales en la arquitectura:

1. **Capa de Atención:**
   - Se añadió una capa de atención sobre la salida de las últimas capas convolucionales para dar mayor peso a las regiones clave del espectrograma.

2. **Reducción de Complejidad:**
   - Se redujo la cantidad de parámetros de las capas densas finales para mejorar la generalización en un dataset pequeño como ESC-50.

3. **Entrenamiento Progresivo:**
   - Comenzó con un dataset reducido y se incrementó gradualmente la cantidad de datos tras cada iteración para estabilizar el aprendizaje.

---

### Evaluación y Validación Cruzada

#### Proceso de Validación
Se empleó una estrategia de validación cruzada basada en los folds del dataset ESC-50, asegurando una cobertura uniforme de las clases:

1. **Configuración de los Folds:**
   - ESC-50 ya contiene una división en cinco folds. Se utilizó esta estructura para asegurar la independencia entre los conjuntos de entrenamiento y validación.

2. **Evaluación por Fold:**
   - Cada modelo fue entrenado y validado cinco veces (una por fold) y las métricas se promediaron al final para obtener resultados representativos.

#### Visualización y Métricas Avanzadas
Se diseñaron visualizaciones detalladas para comprender el desempeño del modelo:

1. **Heatmaps de Atención:**
   - Mapas que mostraban las áreas del espectrograma que el modelo consideraba más relevantes para la clasificación.
   - Ejemplo:
     ```python
     import matplotlib.pyplot as plt
     plt.imshow(attention_weights, cmap='viridis')
     plt.show()
     ```

2. **Análisis de Robustez:**
   - Se evaluó cómo los modelos manejaban datos con condiciones adversas, como ruido extremo o velocidad alterada.

3. **Gráficos de Comparación:**
   - Comparativas entre modelos propios y arquitecturas preentrenadas usando métricas como la precisión promedio por clase.

---

### Conclusiones y Futuras Direcciones

#### Lecciones Aprendidas
1. **Preprocesamiento Esencial:**
   - La calidad del preprocesamiento tuvo un impacto directo en el rendimiento del modelo. La normalización adecuada y la generación de espectrogramas consistentes fueron claves para garantizar resultados replicables.

2. **Importancia del Augmentation:**
   - Incrementar la variabilidad de los datos a través de augmentaciones específicas permitió mejorar la generalización sin requerir un aumento en el tamaño del dataset.

3. **Modelos Especializados:**
   - Las arquitecturas diseñadas específicamente para espectrogramas, como AST, mostraron una clara ventaja frente a modelos generales como ResNet50.

#### Propuestas Futuras
1. **Ampliación del Dataset:**
   - Integrar datos adicionales de otras fuentes para enriquecer las clases existentes.

2. **Exploración de Transformers:**
   - Investigar más a fondo el uso de modelos basados en transformers, que han demostrado un potencial significativo en datos secuenciales.

3. **Optimización en Aceleradores:**
   - Ajustar el pipeline para aprovechar al máximo los recursos de hardware como GPUs y TPUs, reduciendo el tiempo de entrenamiento.
