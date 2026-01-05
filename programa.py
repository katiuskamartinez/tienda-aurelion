import sys
import webbrowser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import webbrowser
from pathlib import Path


def analizar_regresion_importe():
    """
    Realiza un modelo de Regresión Lineal para predecir el Importe de Venta,
    calcula métricas y muestra el gráfico de dispersión.
    """
    salida = "🚀 MODELO DE REGRESIÓN: Predicción del Importe de Venta 🚀\n"
    salida += "="*60 + "\n"
    
    try:
        # Cargar datos
        df = pd.read_excel('Assets/detalle_ventas.xlsx')
        
        # 1. Preparar los datos
        X = df[['cantidad', 'precio_unitario']].fillna(0) # Características
        y = df['importe'] # Etiqueta (variable a predecir)
        
        # 2. Dividir y entrenar el modelo
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        modelo_regresion = LinearRegression()
        modelo_regresion.fit(X_train, y_train)
        
        # 3. Generar predicciones y calcular métricas
        y_pred = modelo_regresion.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        salida += "RESULTADOS DE LAS MÉTRICAS:\n"
        salida += f"- Error Absoluto Medio (MAE): ${mae:.2f} (Error promedio en la predicción)\n"
        salida += f"- Raíz del Error Cuadrático Medio (RMSE): ${rmse:.2f} (Penaliza errores grandes)\n"
        
        # 4. Representación Gráfica
        plt.figure(figsize=(9, 7))
        plt.scatter(y_test, y_pred, alpha=0.6, label='Predicciones')
        # Línea de perfección (y=x)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Predicción Perfecta') 
        plt.xlabel("Importe Real ($)")
        plt.ylabel("Importe Predicho ($)")
        plt.title("Modelo de Regresión: Importe Real vs. Predicho")
        plt.legend()
        plt.grid(True)
        
        salida += "\n✅ Gráfico de dispersión generado en ventana externa.\n"
        plt.show()

        return salida

    except FileNotFoundError:
        return "\n⚠️ ERROR: Archivo 'Assets/detalle_ventas.xlsx' no encontrado. No se puede ejecutar el ML."
    except Exception as e:
        return f"\n❌ ERROR al ejecutar el modelo de Regresión: {e}"

def generar_tres_graficos_representativos():
    """
    Genera y muestra secuencialmente tres gráficos clave (Barras, Dispersión, Box Plot) 
    en ventanas externas y devuelve el resumen textual.
    """
    salida = "📊 GRÁFICOS REPRESENTATIVOS DEL NEGOCIO 📊\n"
    salida += "="*50 + "\n"
    
    try:
        # --- Gráfico 1: Distribución de Clientes por Ciudad (Bar Plot) ---
        df_clientes = pd.read_excel('Assets/clientes.xlsx')
        conteo_clientes_por_ciudad = df_clientes['ciudad'].value_counts()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x=conteo_clientes_por_ciudad.index, 
            y=conteo_clientes_por_ciudad.values,
            hue=conteo_clientes_por_ciudad.index, 
            palette="viridis",
            legend=False
        )
        plt.title('1. Distribución de Clientes por Ciudad (Conteo)', fontsize=16)
        plt.xlabel('Ciudad', fontsize=12)
        plt.ylabel('Número de Clientes', fontsize=12)
        plt.xticks(rotation=45, ha='right') 
        for index, value in enumerate(conteo_clientes_por_ciudad.values):
            plt.text(index, value + 0.5, str(value), ha='center')
        plt.tight_layout()
        salida += "✅ **GRÁFICO 1 ABIERTO:** Distribución de Clientes por Ciudad.\n"
        plt.show() 
        
        # --- Gráfico 2: Relación Cantidad vs. Importe (Reg Plot) ---
        df_ventas = pd.read_excel('Assets/detalle_ventas.xlsx')

        plt.figure(figsize=(10, 7))
        sns.regplot(
            x='cantidad', y='importe', data=df_ventas,
            scatter_kws={'alpha':0.3, 's':20},
            line_kws={'color':'red', 'linewidth':2}
        )
        plt.title('2. Relación entre Cantidad y Importe Total', fontsize=16)
        plt.xlabel('Cantidad de Productos', fontsize=14)
        plt.ylabel('Importe Total de Venta', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        salida += "✅ **GRÁFICO 2 ABIERTO:** Relación Cantidad vs. Importe.\n"
        plt.show() 

        # --- Gráfico 3: Distribución Precio Unitario por Categoría (Box Plot) ---
        # ATENCIÓN: Se requiere el archivo 'productos_corregidos.xlsx'
        df_productos = pd.read_excel("Assets/productos_corregidos.xlsx")

        plt.figure(figsize=(14, 8))
        sns.boxplot(
            x='categoria', y='precio_unitario', data=df_productos,
            palette="pastel", hue='categoria', legend=False
        )
        plt.title('3. Distribución del Precio Unitario por Categoría', fontsize=18, fontweight='bold')
        plt.xlabel('Categoría', fontsize=14)
        plt.ylabel('Precio Unitario ($)', fontsize=14)
        plt.xticks(rotation=45, ha='right') 
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        salida += "✅ **GRÁFICO 3 ABIERTO:** Distribución del Precio Unitario por Categoría.\n"
        plt.show()
        
        return salida
    
    except FileNotFoundError as e:
        return f"\n⚠️ ERROR: No se encontró uno de los archivos Excel necesarios en 'Assets/'. Revise: {e}"
    except Exception as e:
        return f"\n❌ ERROR al generar los gráficos representativos: {e}"

def analizar_outliers():
    """
    Genera el Box Plot para la detección de outliers en 'importe' 
    y calcula el rango intercuartílico (IQR) para encontrar valores atípicos.
    """
    salida = "🔍 ANÁLISIS DE VALORES ATÍPICOS (OUTLIERS) 🔍\n"
    salida += "="*50 + "\n"
    
    try:
        # Cargar el archivo
        df = pd.read_excel('Assets/detalle_ventas.xlsx')

        # --- Gráfico 1: Box Plot (Visualización) ---
        plt.figure(figsize=(6, 8))
        sns.boxplot(y=df['importe'])
        plt.title('Box Plot para detección de Outliers en Importe', fontsize=14)
        
        salida += "✅ **BOX PLOT ABIERTO:** Se ha generado el Box Plot del 'importe' en una ventana externa.\n"
        plt.show() # Muestra la gráfica
        
        # --- Cálculo y Filtrado de Outliers (Método IQR) ---
        
        # Nota: La lógica original usa Q1 de 'precio_unitario' y Q3 de 'importe', 
        # pero para el cálculo de un solo conjunto de límites, usaremos la misma columna para Q1 y Q3.
        # Ajustaremos la lógica para el 'importe' (que es la columna graficada) para mantener la coherencia.
        
        Q1 = df['importe'].quantile(0.25)
        Q3 = df['importe'].quantile(0.75)
        IQR = Q3 - Q1

        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        # Filtrar los outliers basados en los límites de 'importe'
        outliers = df[
            (df['importe'] < limite_inferior) | 
            (df['importe'] > limite_superior)
        ]
        
        salida += "\n--- Análisis IQR de la Columna 'importe' ---\n"
        salida += f"Q1 (25%): {Q1:.2f}\n"
        salida += f"Q3 (75%): {Q3:.2f}\n"
        salida += f"Rango Intercuartílico (IQR): {IQR:.2f}\n"
        salida += f"Límite Inferior: {limite_inferior:.2f}\n"
        salida += f"Límite Superior: {limite_superior:.2f}\n"
        
        salida += "\nValores Atípicos Encontrados (Outliers):\n"
        if outliers.empty:
            salida += "-> No se encontraron outliers utilizando el método IQR para la columna 'importe'."
        else:
            salida += outliers.to_string()

        return salida
    
    except FileNotFoundError:
        return "\n⚠️ ERROR: No se encontró el archivo 'Assets/detalle_ventas.xlsx'. No se pudo realizar el análisis de outliers."
    except Exception as e:
        return f"\n❌ ERROR al analizar outliers: {e}"

def generar_analisis_correlacion():
    """
    Calcula la correlación entre variables, muestra el Gráfico Q-Q 
    y el Mapa de Calor en ventanas emergentes, y devuelve el resumen textual.
    """
    salida = "📈 ANÁLISIS DE CORRELACIÓN Y NORMALIDAD 📈\n"
    salida += "="*50 + "\n"
    
    try:
        # Cargar el archivo
        df = pd.read_excel('Assets/detalle_ventas.xlsx', engine='openpyxl')
        
        # 1. Correlación entre Cantidad e Importe
        correlacion_cant_importe = df['cantidad'].corr(df['importe'])
        salida += f"1. Correlación de Pearson (Cantidad vs. Importe): {correlacion_cant_importe:.3f}\n"
        
        if abs(correlacion_cant_importe) > 0.7:
            salida += "   -> Fuerte correlación positiva o negativa.\n"
        elif abs(correlacion_cant_importe) > 0.3:
            salida += "   -> Correlación moderada.\n"
        else:
            salida += "   -> Correlación débil o nula.\n"

        # 2. Matriz de Correlación
        matriz_correlacion = df[['cantidad', 'precio_unitario', 'importe']].corr()
        salida += "\n2. Matriz de Correlación (Valores):\n"
        salida += matriz_correlacion.to_string() + "\n"
        
        # --- Gráfico 1: Gráfico Q-Q (Prueba de Normalidad) ---
        columna_analizar = 'importe'
        plt.figure(figsize=(8, 6))
        stats.probplot(df[columna_analizar], dist="norm", plot=plt)
        plt.title(f'Gráfico Q-Q para {columna_analizar}', fontsize=14)
        plt.grid(True)
        
        salida += "\n" + "-"*50 + "\n"
        salida += "✅ **GRÁFICO Q-Q ABIERTO:** Se ha generado y abierto la prueba de normalidad en una ventana externa.\n"
        
        plt.show() # Muestra el primer gráfico (Q-Q)

        # --- Gráfico 2: Mapa de Calor de Correlación ---
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            matriz_correlacion,
            annot=True,      # Muestra los valores numéricos
            fmt=".2f",       # Formatea los números con dos decimales
            cmap='coolwarm', # Esquema de color
            cbar=True        # Muestra la barra de color
        )
        plt.title('Mapa de Calor de la Matriz de Correlación', fontsize=14)
        
        salida += "\n" + "-"*50 + "\n"
        salida += "✅ **MAPA DE CALOR ABIERTO:** Se ha generado el mapa de calor en una segunda ventana externa.\n"
        
        plt.show() # Muestra el segundo gráfico (Mapa de Calor)
        
        return salida
    
    except FileNotFoundError:
        return "\n⚠️ ERROR: No se encontró el archivo 'Assets/detalle_ventas.xlsx'. No se pudo generar el análisis de correlación."
    except Exception as e:
        return f"\n❌ ERROR al generar el análisis de correlación: {e}"

def generar_analisis_estadistico():
    """Ejecuta el análisis de datos de clientes y ventas y devuelve el resultado como una cadena."""
    
    salida = "📊 ANÁLISIS ESTADÍSTICO COMPLETO 📊\n"
    salida += "="*50 + "\n\n"
    
    try:
        # --- 1. ANÁLISIS DE CLIENTES (clientes.xlsx) ---
        salida += "⭐ ANÁLISIS DE LA TABLA 'CLIENTES' ⭐\n"
        salida += "-"*40 + "\n"
        df_clientes = pd.read_excel("Assets/clientes.xlsx")
        
        # 1. Conteo de clientes por ciudad
        conteo_clientes_por_ciudad = df_clientes['ciudad'].value_counts()
        salida += "1. Clientes por Ciudad (Conteo):\n"
        salida += conteo_clientes_por_ciudad.to_string() + "\n\n"
        
        # 2. Porcentaje de clientes por ciudad
        frecuencia_relativa_ciudad = df_clientes['ciudad'].value_counts(normalize=True) * 100
        salida += "2. Porcentaje de Clientes por Ciudad:\n"
        salida += frecuencia_relativa_ciudad.round(2).to_string() + " %\n\n"
        
        salida += "\n" + "="*50 + "\n\n"
        
        # --- 2. ANÁLISIS DE VENTAS (detalle_ventas.xlsx) ---
        salida += "💰 ANÁLISIS DESCRIPTIVO DE LA TABLA 'VENTAS' 💰\n"
        salida += "-"*40 + "\n"
        
        detalle_ventas = pd.read_excel('Assets/detalle_ventas.xlsx')
        
        # Diccionario para almacenar los resultados de ventas
        resultados_ventas = {}

        for columna in ['cantidad', 'importe']:
            # Convertir a numérico y manejar errores
            detalle_ventas[columna] = pd.to_numeric(detalle_ventas[columna], errors='coerce')
            
            # Obtener datos válidos
            datos_validos = detalle_ventas[columna].dropna()

            if not datos_validos.empty:
                resultados_ventas[columna] = {
                    'Media': datos_validos.mean(),
                    'Mediana': datos_validos.median(),
                    'Moda': datos_validos.mode().tolist(),
                    'Desviación Estándar': datos_validos.std()
                }
            else:
                resultados_ventas[columna] = "No hay datos numéricos válidos para calcular estadísticos."

        # Formatear la salida de los resultados de ventas
        for columna, stats in resultados_ventas.items():
            salida += f"\n📈 **Estadísticas para la Columna: {columna.upper()}**\n"
            if isinstance(stats, str):
                salida += f"  {stats}\n"
            else:
                salida += f"  **Media (Promedio):** {stats['Media']:.2f}\n"
                salida += f"  **Mediana (Valor Central):** {stats['Mediana']:.2f}\n"
                # Se imprime la moda de forma clara
                moda_str = ', '.join(map(str, stats['Moda']))
                salida += f"  **Moda(s) (Más Frecuente):** {moda_str}\n"
                salida += f"  **Desviación Estándar (Std):** {stats['Desviación Estándar']:.2f}\n"
        
        return salida
    
    except FileNotFoundError:
        return "\n⚠️ ERROR: No se encontraron los archivos Excel en la carpeta 'Assets/'. Asegúrate de que existan 'clientes.xlsx' y 'detalle_ventas.xlsx'."
    except Exception as e:
        return f"\n❌ ERROR al ejecutar el análisis estadístico: {e}"

def generar_distribucion_grafica(columna_analizar='importe'):
    """
    Realiza el análisis de distribución, muestra el gráfico en una ventana emergente
    y devuelve el resumen textual.
    """
    salida = f"📊 ANÁLISIS DE DISTRIBUCIÓN (Columna: {columna_analizar.upper()}) 📊\n"
    salida += "="*50 + "\n"
    
    try:
        # Cargar el archivo
        df = pd.read_excel('Assets/detalle_ventas.xlsx')

        # Calcular Estadísticos Clave
        media = df[columna_analizar].mean()
        mediana = df[columna_analizar].median()
        asimetria = df[columna_analizar].skew()
        
        # --- Resumen Textual ---
        salida += f"1. Media de '{columna_analizar}': {media:.2f}\n"
        salida += f"2. Mediana de '{columna_analizar}': {mediana:.2f}\n"
        salida += f"3. Asimetría (Skewness): {asimetria:.2f}\n"

        # Interpretación de la Asimetría
        if abs(asimetria) < 0.5:
            salida += "   -> Distribución probablemente Simétrica (Normal o Uniforme).\n"
        elif asimetria > 0.5:
            salida += "   -> Distribución con Asimetría Positiva (Sesgada a la Derecha).\n"
        else: # asimetria < -0.5
            salida += "   -> Distribución con Asimetría Negativa (Sesgada a la Izquierda).\n"

        # --- Generación y Visualización del Gráfico ---
        plt.figure(figsize=(10, 6))
        sns.histplot(df[columna_analizar], kde=True, bins=30)
        plt.title(f'Histograma y Densidad de {columna_analizar}', fontsize=14)
        plt.xlabel(columna_analizar, fontsize=12)
        plt.ylabel('Frecuencia', fontsize=12)
        
        # Líneas de referencia
        plt.axvline(media, color='red', linestyle='dashed', linewidth=1.5, label=f'Media ({media:.2f})')
        plt.axvline(mediana, color='green', linestyle='dashed', linewidth=1.5, label=f'Mediana ({mediana:.2f})')
        plt.legend()
        
        salida += "\n" + "-"*50 + "\n"
        salida += "✅ **GRÁFICO ABIERTO:** Se ha generado y abierto el Histograma en una ventana externa.\n"
        
        # **LÍNEA CLAVE:** Muestra la gráfica en una ventana nueva y bloquea el programa brevemente
        plt.show() 
        
        return salida
    
    except FileNotFoundError:
        return "\n⚠️ ERROR: No se encontró el archivo 'Assets/detalle_ventas.xlsx'. No se pudo generar la gráfica."
    except Exception as e:
        return f"\n❌ ERROR al generar el gráfico de distribución: {e}"

def abrir_archivo_externo(nombre_archivo):
    """Intenta abrir un archivo local en una aplicación externa (navegador/visor)."""
    try:
        # webbrowser.open() es el método más compatible para intentar abrir archivos
        # o URLs en el visor/navegador predeterminado del sistema operativo.
        webbrowser.open(nombre_archivo)
        return f"\n✅ Intentando abrir '{nombre_archivo}' en la aplicación predeterminada de su sistema..."
    except Exception as e:
        return f"\n❌ No se pudo abrir el archivo '{nombre_archivo}' automáticamente. Error: {e}"

def formato_tablas_principales():
    """Formatea la tabla de la base de datos con mejor alineación."""
    datos = [
        ("Tabla", "Campos"),
        ("Clientes", "Id, Nombre, Email, Ciudad, Fecha de alta"),
        ("Detalles Ventas", "Id de la venta, Id del producto, Cantidad, Precio unitario, Importe"),
        ("Productos", "Id del producto, Nombre del producto, Categoría, Precio unitario"),
        ("Ventas", "Id, Fecha, Id del cliente, Nombre del cliente, Email, Medio de pago")
    ]
    
    # Definir anchos para cada columna
    ancho_tabla = 20
    ancho_campos = 70
    
    separador = f"+{'-'*ancho_tabla}+{'-'*ancho_campos}+\n"
    salida = "\n### Tablas Principales\n"
    salida += separador
    
    # Imprimir encabezado
    encabezado = f"| {'Tabla'.ljust(ancho_tabla-1)}| {'Campos'.ljust(ancho_campos-1)}|\n"
    salida += encabezado
    salida += separador
    
    # Imprimir filas de datos
    for i, (tabla, campos) in enumerate(datos):
        if i == 0: continue # Saltar el encabezado que ya fue impreso
        
        # El .ljust() asegura que el texto tenga el ancho definido
        fila = f"| {tabla.ljust(ancho_tabla-1)}| {campos.ljust(ancho_campos-1)}|\n"
        salida += fila

    salida += separador
    
    return salida

# 1. Almacenamiento de la Documentación en una estructura de datos (Diccionario)
#    Cada clave es el número del paso, y el valor es el contenido formateado.
documentacion = {
    "1": {
        "titulo": "Tema",
        "contenido": (
            "**Generar la Documentación Técnica Para ver cual es la Estructura de la base de Datos, "
            "Tipos de datos y tablas Principales. Para crear un Menu interactivo para Tienda tipo Minimarket (Tienda Aurelion)**"
        )
    },
    "2": {
        "titulo": "Problema",
        "contenido": (
            "**Tienda Aurelion no tiene soluciones tecnológicas para poder visualizar su documentación.**"
        )
    },
    "3": {
        "titulo": "Solución",
        "contenido": (
            "**Armado de Documentación Técnica y creación de chatbot intectativo para visualizar "
            "las diferentes opciones de la documentación**"
        )
    },
    "4": {
        "titulo": "Estructura de la Base de Datos",
        "contenido": (
            "## Fuente\n"
            "Bases de Datos suministradas por Tienda Aurelion\n"

            "## Tipo de Base de Datos\n"
            "Base de datos Relacional\n"

            "## Tipo de Datos\n"
            "Datos Estructurados\n"

            "## Tipo de los datos de campo:\n"
            "id_venta, id_producto: Numérico/Entero\n"
            "nombre_producto: Texto/Cadena.\n"
            "cantidad: Numérico/Entero.\n"
            "precio_unitario, importe: Numérico/Moneda\n"
           
           + formato_tablas_principales() +
            "\n### Cantidad de Registros\n"
            "  - **Ventas:** 120 registros\n"
            "  - **Detalles Ventas:** 120 registros\n"
            "  - **Productos:** 100 productos registrados\n"
            "  - **Clientes:** 100 registros de clientes\n"

            "\n## Escala de Medición\n"
            "- **Nominal:** El tipo de producto está en una categoría sin orden.\n"
            "- **Intervalo:** Fechas de las ventas y altas de clientes.\n"
            "- **Razón:** Ingresos y cantidad de ventas."
        )
    },
   "5": {
        "titulo": "Instrucciones Sugeridas para el Asistente de IA",
        "contenido": (
            "\n"
            "# 📝 Instrucciones para el Asistente de IA (Contexto: Tienda Aurelion Minimarket)\n"
            "---------------------------------------------------\n"
            
            "## 1. Contexto del Proyecto\n"
            "  - **Tema (Paso 1):** Generar la Documentación Técnica y un Menú Interactivo tipo Chatbot para Tienda Aurelion, visualizando la estructura de la base de datos, tipos de datos y tablas principales.\n"
            "  - **Problema (Paso 2):** Tienda Aurelion carece de soluciones tecnológicas para visualizar y acceder a su documentación técnica.\n"
            "  - **Solución (Paso 3):** Creación de la Documentación Técnica formal y un chatbot interactivo para consultarla.\n"
            
            "\n## 2. Pasos y Tareas Iniciales para el Asistente de IA\n"
            "El asistente debe realizar las siguientes tareas para asegurar la funcionalidad del chatbot y la coherencia del proyecto:\n"
            "  - **Paso:** Asegurar que el chatbot maneje correctamente los mensajes descriptivos (Paso 6 y 7).\n"
            "  - **Tarea:** Crear un archivo llamado `pseudocodigo.ipynb` con el pseudocódigo simple del chatbot.\n"
            "  - **Tarea:** Generar Script de Inicialización (`chatbot_doc.py`) que contenga todas las funciones (`mostrar_menu`, `iniciar_chatbot`) y el diccionario `documentacion` actualizado.\n"
            
            "\n## 3. Sugerencias para el Desarrollo y Usabilidad\n"
            "### Sugerencias de Código (Python)\n"
            "  - **Robustez de Entrada:** En `iniciar_chatbot()`, añadir manejo explícito para entrada vacía (`\"\"`) para evitar fallos.\n"
            "  - **Manejo de Tildes:** Considerar normalizar las entradas del usuario (ej. eliminar tildes/caracteres especiales) para una búsqueda más flexible.\n"
            "  - **Diagrama ERD:** Recomendar generar la imagen `diagrama_de_flujo.png` como un Diagrama Entidad-Relación (ERD) que muestre las relaciones entre las cuatro tablas principales.\n"
            
            "\n## 4. Mejoras Futuras para el Chatbot (IA)\n"
            "| Tipo de Mejora | Descripción |\n"
            "| :--- | :--- |\n"
            "| **Interacción** | Permitir que el usuario ingrese el título (ej. \"Estructura\") para acceder al paso. |\n"
            "| **Búsqueda** | Implementar una función de búsqueda por palabras clave. |\n"
            "| **Exportación** | Añadir una opción para exportar el contenido de un paso a un archivo de texto (`.txt`) local. |\n"
            "| **Integración DB** | (Largo Plazo) Permitir que el chatbot lea datos de configuración (`.json` o `.ini`) en lugar de tenerlos codificados en Python. |\n"
        )
    },
    "6": {
       "titulo": "Pseudocódigo",
        "archivo": "pseudocodigo.ipynb",  # Nombre del archivo a abrir
        "contenido": "Presione 6 para intentar abrir el archivo de pseudocódigo en su visor de Notebooks predeterminado."
    },
    "7": {
        "titulo": "Diagrama de Flujo",
        "archivo": "diagrama_de_flujo.png", # Nombre del archivo a abrir
        "contenido": "Presione 7 para intentar abrir la imagen del diagrama de flujo en su visor de imágenes predeterminado."
    },
    "8": {
        "titulo": "Estadísticas de Datos (Pandas)",
        "contenido": generar_analisis_estadistico()
    },
    "9": {
        "titulo": "Distribución de Datos (Gráfica Externa)",
        "contenido": "Presione 9 para generar y visualizar la distribución de la columna 'importe' en una ventana gráfica externa."
    },
    "10": {
        "titulo": "Correlación y Gráficos Q-Q/Heatmap",
        "contenido": "Presione 10 para generar el análisis de correlación, la Matriz de Correlación y los gráficos Q-Q/Mapa de Calor en ventanas externas."
    },
    "11": {
        "titulo": "Análisis y Detección de Outliers (Box Plot)",
        "contenido": "Presione 11 para realizar el análisis de outliers usando el método IQR y visualizar el Box Plot del 'importe' en una ventana externa."
    },
    "12": {
        "titulo": "Interpretación y Hallazgos de Resultados",
        "contenido": (
            "\n"
            "# 5 — Interpretación de Resultados\n"
            "---------------------------------------------------\n"
            
            "A continuación se resumen e interpretan los hallazgos principales derivados de los gráficos generados.\n"
            
            "\n## 1) Distribución de clientes por ciudad\n"
            " - **Observación:** Se identifican ciudades con concentración alta de clientes y otras con muy pocos.\n"
            " - **Interpretación:** La demanda está focalizada; puede indicar foco de operaciones o desigualdad en la cobertura.\n"
            " - **Acción Recomendada:** Priorizar acciones comerciales y logística en ciudades con mayor volumen; investigar causas de baja representación en otras.\n"

            "\n## 2) Relación cantidad de productos vendidos vs. importe total\n"
            " - **Observación:** Existe una tendencia positiva (a mayor cantidad, mayor importe) pero con dispersión y outliers.\n"
            " - **Interpretación:** Los outliers (importe alto con baja cantidad) pueden indicar productos de alto precio o errores/errores de ingreso.\n"
            " - **Acción Recomendada:** Calcular la correlación (Pearson) y R²; revisar ventas atípicas para detectar precios erróneos o transacciones especiales.\n"

            "\n## 3) Distribución del precio unitario por categoría\n"
            " - **Observación:** Diferencias en medianas y rangos intercuartílicos entre categorías; presencia de valores extremos (outliers).\n"
            " - **Interpretación:** Algunas categorías tienen mayor variabilidad de precio, otras están más homogéneas.\n"
            " - **Acción Recomendada:** Segmentar precios por subcategoría, revisar outliers (posibles errores o SKUs premium), y calibrar estrategia de precios.\n"

            "\n## Limitaciones\n"
            " - **Calidad y completitud:** Valores faltantes, formatos de fecha o errores en ingreso pueden sesgar las gráficas.\n"
            " - **Contexto:** Faltan variables clave (margen, costo, promociones, canal de venta) para conclusiones comerciales firmes.\n"

            "\n## Próximos pasos sugeridos\n"
            " 1. Calcular métricas cuantitativas: correlaciones, R², estadísticas por grupo.\n"
            " 2. Investigar outliers y limpiar datos (fechas inválidas, importes negativos, cantidades inconsistentes).\n"
            " 3. Enriquecer datos con variables adicionales (canal, costo, fecha de campaña) y repetir análisis.\n"
            " 4. Crear un dashboard interactivo para monitorizar las métricas clave por ciudad y categoría.\n"
            
            "\n(El chatbot puede generar celdas de código para calcular las correlaciones, detectar outliers y producir tablas resumen automáticamente.)\n"
        )
    },
    "13": {
        "titulo": "Tres Gráficos Representativos (Bar, Dispersión, Box)",
        "contenido": "Presione 13 para generar y visualizar tres gráficos clave del negocio en ventanas externas."
    },
    "14": {
        "titulo": "Modelo ML: Regresión para Predicción de Importe",
        "contenido": "Presione 14 para entrenar el modelo de Regresión Lineal y ver el error de predicción y el gráfico de dispersión."
    },
    "15": {
        'titulo': "Power BI: Tienda Aurelion. Introducción General y Resumen Ejecutivo",
         "archivo": "Proyecto Aurelión Power BI.pbix",  # Nombre del archivo a abrir
        'contenido': """
### Introducción: Análisis de Datos de la Tienda Aurelion
El objetivo de este proyecto fue transformar datos transaccionales en inteligencia de negocio utilizable, sentando las bases para la toma de decisiones estratégicas.

1.  **Modelo de Datos Centralizado:** Unificamos las cuatro fuentes de datos (Ventas, Clientes, Productos, Calendario) en un **Modelo de Estrella** optimizado para análisis multidimensional.
2.  **KPIs y Análisis de Escenarios:** Creamos **KPIs de rendimiento** y métricas de **Time Intelligence** (continuidad de datos, metas) para monitorear la actividad del cliente y el precio promedio de venta.
3.  **Evaluación de Rendimiento:** Validamos que la fórmula de nuestro modelo de regresión es altamente precisa, permitiendo una medición clara de errores y anomalías (outliers).

  "Estructura de Datos y Modelo Estrella",
El reporte se basa en un modelo de estrella con las siguientes tablas:

+----------------+------------+------------------------------------------+-------------+
|     Tabla      |    Tipo    |               Descripción                |    Clave    |
+----------------+------------+------------------------------------------+-------------+
| FactVentas     | Hechos     | Todas las transacciones de venta         | id_venta    |
| DimClientes    | Dimensión  | Información de los clientes (Ciudad, ID) | id_cliente  |
| DimProductos   | Dimensión  | Datos de los productos (Categoría, Precio| id_producto |
| DimCalendario  | Dimensión  | Fechas continuas para análisis de tiempo | Date        |
+----------------+------------+------------------------------------------+-------------+

Relaciones:
- DimClientes   (1) ───► (*) FactVentas   [id_cliente]
- DimProductos  (1) ───► (*) FactVentas   [id_producto]
- DimCalendario (1) ───► (*) FactVentas   [Date]



"Métricas Clave (KPIs)",
        
### A. KPI 2: Frecuencia de Compra vs. Meta (Actividad)
- Indicador: **[Conteo Transacciones]** (`DISTINCTCOUNT(FactVentas[id_venta])`)
- Objetivo: **[Meta Frecuencia]** (20 - Asumido Mensual)

### B. KPI 3: Rendimiento de Precio Promedio por Ciudad
- Indicador: **[Precio Promedio x Unidad]** (`DIVIDE([Ventas Totales], [Cantidad Vendida], 0)`)
- Objetivo: **[Umbral Precio Promedio]** (1000)

### C. Lealtad y Recurrencia
- Indicador: **[% Clientes Recurrentes]**
- Lógica DAX Avanzada: Se usa el conteo de transacciones por cliente para filtrar aquellos con más de una compra.

    ```dax
    Clientes Recurrentes = 
        COUNTROWS(
            FILTER(
                VALUES(FactVentas[id_cliente]), 
                CALCULATE(DISTINCTCOUNT(FactVentas[id_venta])) > 1
            )
        )
    ```
### D. Visualizaciones Clave
    "Visualizaciones Clave y Diagnóstico",
       
### 1. Gráfico de Dispersión (Outliers)
- **Ejes:** Eje X: [precio_unitario], Eje Y: [importe]. Detalles: [id_venta].
- **Propósito:** Identificar transacciones atípicas (outliers) y puntos con alto error respecto a la línea de regresión. Se usa la opción **"No resumir"** en los ejes.

### 2. Evolución del Medio de Pago
- **Visual:** Gráfico de Área Apilada 100%.
- **Propósito:** Monitorear el cambio en la proporción de ingresos generados por Efectivo, Tarjeta y QR a lo largo del tiempo, crucial para el análisis de costos de transacción.

### 3. Continuidad Temporal
- **Configuración:** Todos los gráficos de tendencia usan `DimCalendario[Date]` configurada como **Fecha Continua** para evitar interrupciones en la serie de tiempo.
      
"Simulación de la Captura de Presentación",
      
La siguiente es una representación esquemática del diseño de la página principal del informe, enfocada en los KPIs clave:

+-----------------------------------------------------------+
|               DASHBOARD PRINCIPAL TIENDA AURELION         |
|-----------------------------------------------------------|
| KPI 2: FRECUENCIA     | KPI 3: PRECIO PROM. | KPI: REC.   |
| 500 TRANSACCIONES     | $3,500 / UNIDAD     | 28% REC.    |
| META: 100 (MES)       | META: $3,000        | META: 30%   |
| [Gráfico Tendencia]   | [Gráfico Tendencia] | [Tendencia] |
|-----------------------------------------------------------|
| 📈 Tendencia Ventas por Categoría (Área Apilada)          |
| [Gráfico mostrando las franjas de Ingresos por Categoría]  |
|-----------------------------------------------------------|
| 🏙️ Concentración Clientes | 💳 Evolución Medio de Pago     |
| [Barras por Ciudad]      | [Área Apilada 100% por Pago]   |
+-----------------------------------------------------------+
### El análisis completo está disponible en el informe Power BI.     
        """
    },
}

# 2. Función para mostrar el menú principal
def mostrar_menu():
    """Muestra el menú con las opciones disponibles."""
    print("\n" + "="*50)
    print("🤖 Chatbot de Documentación Técnica Tienda Aurelion 📊")
    print("="*50)
    print("Por favor, selecciona el número del paso que deseas revisar:")
    
    # Itera sobre las claves y valores del diccionario para listar los títulos
    for key, item in documentacion.items():
        print(f"[{key}] - {item['titulo']}")
        
    print("[M] - Mostrar este menú")
    print("[S] - Salir del Chatbot")
    print("="*50)

# 3. Función principal del Chatbot
def iniciar_chatbot():
    """Inicia el bucle principal de interacción del chatbot."""
    mostrar_menu()
    
    while True:
        entrada = input("👉 Ingresa tu opción (1-7, M, S): ").upper().strip()
        
        if entrada == 'S':
            print("\n👋 ¡Gracias por usar el Chatbot! 👋")
            sys.exit()
        elif entrada == 'M':
            mostrar_menu()
            continue
            
        elif entrada in documentacion:
            paso = documentacion[entrada]
            print("\n" + "#"*50)
            print(f"PASO {entrada}: {paso['titulo']}")
            print("#"*50)

            # Lógica para llamar a funciones si el contenido es una función (ej. Paso 8  9 10 11 13 14)
    # -----------------------------------------------------
            if entrada == '8': # Análisis Estadístico
              print(generar_analisis_estadistico())
            elif entrada == '9': # Gráfico de Distribución
              print(generar_distribucion_grafica())
            elif entrada == '10': # Correlación y Gráficos
                print(generar_analisis_correlacion())
            elif entrada == '11': # Análisis de Outliers
                print(analizar_outliers())
            elif entrada == '13': # Tres Gráficos Representativos
                print(generar_tres_graficos_representativos())
            elif entrada == '14': # Regresión de Importe
                print(analizar_regresion_importe())
    # -----------------------------------------------------
            # --- Lógica para abrir archivo externo (Pasos 6 y 7) ---
            if 'archivo' in paso:
                resultado = abrir_archivo_externo(paso['archivo'])
                print(resultado)
                
                # Imprimir el contenido estático (instrucción) si lo hay
                print(paso['contenido']) 

            # Lógica para contenido estático (Pasos 1, 2, 3, 4, 5 Y 12)
            else:
                print(paso['contenido'])

            print("\n--- Fin del contenido del paso ---")
            
        else:
            print("\n❌ Opción no válida. Por favor, ingresa un número del 1 al 7, 'M' para el menú o 'S' para salir.")
# 4. Ejecución del programa
if __name__ == "__main__":
    try:
        iniciar_chatbot()
    except EOFError:
        # Manejo de excepción común en entornos interactivos (como Ctrl+D o Ctrl+Z)
        print("\n👋 Sesión terminada. ¡Hasta pronto! 👋")
    except Exception as e:
        # Manejo de otras excepciones
        print(f"\nSe ha producido un error inesperado: {e}")

