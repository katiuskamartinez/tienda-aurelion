📝 Instrucciones para el Asistente de IA (Contexto: Tienda Aurelion Minimarket)

## 1. Contexto del Proyecto

### Punto Descripción

Tema (Paso 1) Generar la Documentación Técnica y un Menú Interactivo tipo Chatbot para Tienda Aurelion, visualizando la estructura de la base de datos, tipos de datos y tablas principales.
Problema (Paso 2) Tienda Aurelion carece de soluciones tecnológicas para visualizar y acceder a su documentación técnica.
Solución (Paso 3) Creación de la Documentación Técnica formal y un chatbot interactivo para consultarla.

## 2. Pasos y Tareas Iniciales para el Asistente de IA

El asistente debe realizar las siguientes tareas para asegurar la funcionalidad del chatbot y la coherencia del proyecto.

Pasos
Asegurar que el chatbot maneje correctamente los mensajes descriptivos (Paso 6 y 7).

Tarea: Crear un archivo llamado pseudocodigo.ipynb con el pseudocódigo simple del chatbot.

Generar Script de Inicialización: Producir el código Python (chatbot_doc.py) que contenga todas las funciones ( mostrar_menu, iniciar_chatbot) y el diccionario documentacion actualizado.

## 3. Sugerencias para el Desarrollo y Usabilidad

Sugerencias de Código (Python)
Robustez de Entrada: En la función iniciar_chatbot(), añadir un manejo explícito para la entrada vacía ("") para evitar fallos si el usuario simplemente presiona Enter.

Manejo de Tildes: Si el chatbot se expande, considerar normalizar las entradas del usuario (ej. eliminar tildes o caracteres especiales) para una búsqueda más flexible.

Diagrama ERD: Recomendar generar la imagen diagrama_de_flujo.png no como un flujo, sino como un Diagrama Entidad-Relación (ERD) que muestre las relaciones entre las cuatro tablas principales.

## 4. Mejoras Futuras para el Chatbot (IA)

Tipo de Mejora Descripción
Interacción Permitir que el usuario no solo ingrese el número (1-7), sino también el título (ej. "Estructura") para acceder al paso, aumentando la flexibilidad.
Búsqueda Implementar una función de búsqueda por palabras clave. Si el usuario escribe "precio", el chatbot debe devolver los pasos (ej. 4) donde esa palabra aparece.
Exportación Añadir una opción al menú para exportar el contenido de un paso (ej. Paso 4) a un archivo de texto (.txt) local.
Integración DB (Largo Plazo) Si los datos de la documentación cambian (ej. la cantidad de registros), el chatbot debería poder leer esa información desde un archivo de configuración (.json o .ini) en lugar de tenerla codificada en Python.

## 💻 Prompt para Análisis de Datos y Documentación con Copilot

"Actúa como un analista de datos experto y mi asistente de documentación. Estás trabajando con un DataFrame de ventas de clientes cargado en Python (Pandas) correspondiente a **tienda aurelion**. Tu objetivo es realizar un Análisis Exploratorio de Datos (ventas) completo y documentar cada paso y resultado en el archivo documentacion.md

## Sigue estas 5 fases con las siguientes instrucciones:

1. Calcular Estadísticas Básicas (Visión General)
   Instrucción: Para todas las columnas numéricas clave (importe, cantidad, precio_unitario), utiliza el método .describe() de Pandas. Para la columna categórica principal (Ciudad), calcula el conteo de clientes por ciudad usando .value_counts().

Documentación con Copilot: Documenta la salida de .describe() y value_counts() en una sección titulada "1. Estadísticas Descriptivas Base". Resalta la Media y la Mediana de las variables de ventas para la comparación inicial.

2. Identificar Tipo de Distribución (Visual y Numérico)
   Instrucción: Elige la columna más crítica para el negocio (ej., importe).

Visual: Genera un Histograma y un Diagrama de Caja y Bigotes (Box Plot) usando seaborn para visualizar la forma y los outliers.

Numérico: Calcula la Asimetría (.skew()) y realiza la Prueba de Shapiro-Wilk (usando scipy.stats) para evaluar la Normalidad.

Documentación con Copilot: Crea una sección llamada "2. Análisis de Distribución (Importe)". Incluye la interpretación del Histograma (forma, picos) y la conclusión de la prueba de asimetría y Shapiro-Wilk, indicando si la distribución es Normal, Sesgada, Uniforme o Bimodal.

3. Calcular Correlaciones entre Variables Principales
   Instrucción: Calcula la Matriz de Correlación (Pearson) entre importe, cantidad, y precio_unitario utilizando .corr(). Genera un Mapa de Calor (heatmap) de esta matriz con seaborn.

Documentación con Copilot: Crea una sección "3. Correlación entre Variables Clave". Documenta la matriz y describe verbalmente las tres relaciones más fuertes (positivas o negativas). Por ejemplo: "Existe una correlación de X.XX entre Y y Z, indicando una relación [fuerte/débil, positiva/negativa]".

4. Analizar Outliers (Valores Atípicos)
   Instrucción: Utiliza el método del Rango Intercuartílico (IQR) en la columna importe (o la más sesgada) para calcular los límites superior e inferior. Filtra el DataFrame para mostrar los registros identificados como outliers.

Documentación con Copilot: Crea la sección "4. Detección y Análisis de Outliers". Documenta el número de outliers encontrados y muestra las primeras 5 filas de los outliers. Formula una hipótesis sobre su origen (ej. 'podrían ser grandes compras de clientes corporativos' o 'errores de registro').

5. Interpretar Resultados para el Problema de Negocio
   Instrucción: Sintetiza los hallazgos de los puntos 1 al 4, centrándote en el problema de negocio principal (ej., ¿Qué impulsa las ventas? ¿Dónde están concentrados los clientes?).

## Documentación con Copilot: Crea la sección final "5. Conclusiones y Recomendaciones de Negocio". Genera un resumen de tres puntos clave (uno por cada área):

Concentración de Clientes: Ciudad con más clientes.

Motor de Ventas: La variable con mayor correlación con el importe.

Riesgo/Oportunidad: Implicación de los outliers y de la asimetría en la distribución.
