### 🛒 Tienda Aurelion: Sistema de Documentación y Business Intelligence

Este repositorio contiene la solución tecnológica integral para Tienda Aurelion, un minimarket que buscaba centralizar su documentación técnica y transformar sus datos transaccionales en decisiones estratégicas.

⚠️ Nota importante — Datos sintéticos 🧪
Los datos usados en este proyecto son sintéticos (creados) y no corresponden a información de usuarios reales. No usar para decisiones que requieran datos reales o confidenciales.

## 📑 Tabla de Contenidos

# Resumen del Proyecto

Estructura de Datos

Instalación y Configuración

KPIs y Métricas de Negocio

Chatbot de Documentación (Python)

## Análisis en Power BI

🎯 Resumen del Proyecto
El Problema
Tienda Aurelion carecía de herramientas para visualizar su estructura de datos y monitorear sus metas de venta de forma automatizada, lo que dificultaba la expansión y el control de inventarios/precios.

La Solución
Se desarrolló un ecosistema que incluye:

Documentación Técnica Estructurada: Definición de campos, tipos y escalas de medición.

Chatbot Interactivo: Una interfaz de consola en Python para consulta rápida de la arquitectura del proyecto.

Dashboard de Power BI: Un modelo de estrella con KPIs avanzados y análisis de regresión para detección de anomalías.

## 🏗️ Estructura de Datos

El proyecto utiliza un Modelo de Estrella (Star Schema) para optimizar el rendimiento de las consultas y la claridad del análisis.

# Tablas del Modelo

| Tabla         | Tipo      | Descripción                             | Clave Primaria |
| ------------- | --------- | --------------------------------------- | -------------- |
| FactVentas    | Hechos    | Registro de todas las transacciones.    | id_venta       |
| DimClientes   | Dimensión | Datos demográficos y fechas de alta.    | id_cliente     |
| DimProductos  | Dimensión | Catálogo de productos y categorías.     | id_producto    |
| DimCalendario | Dimensión | Tabla de tiempo para análisis temporal. | Date           |

## ⚙️ Instalación y Configuración

Sigue estos pasos para configurar el entorno local y ejecutar el chatbot:

1. Clonar el repositorio
   Bash

git clone https://github.com/katiuskamartinez/tienda-aurelion.git
cd tienda-aurelion

2. Crear un Entorno Virtual
   Se recomienda el uso de un entorno virtual para mantener las dependencias aisladas.

En Windows:

Bash

python -m venv venv
.\venv\Scripts\activate
En macOS/Linux:

Bash

python3 -m venv venv
source venv/bin/activate

3. Instalar dependencias
   Bash

pip install -r requirements.txt
(Nota: Asegúrate de que tu archivo requirements.txt incluya pandas, matplotlib y seaborn).

📊 KPIs y Métricas de Negocio
Se definieron indicadores clave de rendimiento (KPIs) para medir la salud de la tienda:

Frecuencia de Compra: Meta de 100 transacciones únicas por mes.

Umbral de Precio Promedio: Meta de $3,000 por unidad vendida.

Fórmula DAX: Precio Promedio = COALESCE(DIVIDE([Ventas Totales], [Cantidad Vendida], 0), 0)

Lealtad de Cliente: Porcentaje de clientes recurrentes (con más de una compra).

🤖 Chatbot de Documentación (Python)
Se implementó un script interactivo (programa.py) que permite navegar por la documentación técnica.

Características principales:

Tablas ASCII: Formateo dinámico de tablas para una mejor visualización en consola.

Análisis Estadístico: Integración con Pandas para mostrar promedios e importes en tiempo real.

Robustez: Manejo de entradas inválidas para evitar cierres inesperados.

Bash

# Ejecución

python chatbot_doc.py
📈 Análisis en Power BI
El reporte de Power BI (.pbix) incluye:

Análisis de Outliers: Gráficos de dispersión para detectar transacciones que se desvían de la línea de regresión (precios erróneos o ventas premium).

Evolución de Medios de Pago: Gráfico de áreas apiladas 100% para monitorear el uso de Efectivo, Tarjeta y QR.

Time Intelligence: Análisis de ventas comparando periodos actuales frente a metas fijas.

🛠️ Tecnologías Utilizadas
Lenguajes: Python (Pandas, Matplotlib, Seaborn).

# 🛒 Tienda Aurelion: Sistema de Documentación y Business Intelligence

Este repositorio contiene la solución tecnológica integral para Tienda Aurelion, un minimarket que buscaba centralizar su documentación técnica y transformar sus datos transaccionales en decisiones estratégicas.

## 📑 Tabla de Contenidos

- Resumen del Proyecto
- Estructura de Datos
- Instalación y Configuración
- KPIs y Métricas de Negocio
- Chatbot de Documentación (Python)
- Análisis en Power BI

## 🎯 Resumen del Proyecto

## El Problema

Tienda Aurelion carecía de herramientas para visualizar su estructura de datos y monitorear sus metas de venta de forma automatizada, lo que dificultaba la expansión y el control de inventarios/precios.

## La Solución

Se desarrolló un ecosistema que incluye:

- Documentación Técnica Estructurada: Definición de campos, tipos y escalas de medición.
- Chatbot Interactivo: Una interfaz de consola en Python para consulta rápida de la arquitectura del proyecto.
- Dashboard de Power BI: Un modelo de estrella con KPIs avanzados y análisis de regresión para detección de anomalías.

## 🏗️ Estructura de Datos

El proyecto utiliza un Modelo de Estrella (Star Schema) para optimizar el rendimiento de las consultas y la claridad del análisis.

## Tablas del Modelo

| Tabla         | Tipo      | Descripción                             | Clave Primaria |
| ------------- | --------- | --------------------------------------- | -------------- |
| FactVentas    | Hechos    | Registro de todas las transacciones.    | id_venta       |
| DimClientes   | Dimensión | Datos demográficos y fechas de alta.    | id_cliente     |
| DimProductos  | Dimensión | Catálogo de productos y categorías.     | id_producto    |
| DimCalendario | Dimensión | Tabla de tiempo para análisis temporal. | Date           |

## ⚙️ Instalación y Configuración

Sigue estos pasos para configurar el entorno local y ejecutar el chatbot:

1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/tienda-aurelion.git
cd tienda-aurelion
```

2. Crear un Entorno Virtual

Se recomienda el uso de un entorno virtual para mantener las dependencias aisladas.

En Windows:

```bash
python -m venv venv
.\venv\Scripts\activate
```

En macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Instalar dependencias

```bash
pip install -r requirements.txt
```

(Nota: Asegúrate de que tu archivo requirements.txt incluya pandas, matplotlib y seaborn).

## 📊 KPIs y Métricas de Negocio

Se definieron indicadores clave de rendimiento (KPIs) para medir la salud de la tienda:

- Frecuencia de Compra: Meta de 100 transacciones únicas por mes.
- Umbral de Precio Promedio: Meta de $3,000 por unidad vendida.
- Fórmula DAX: Precio Promedio = COALESCE(DIVIDE([Ventas Totales], [Cantidad Vendida], 0), 0)
- Lealtad de Cliente: Porcentaje de clientes recurrentes (con más de una compra).

## 🤖 Chatbot de Documentación (Python)

Se implementó un script interactivo (programa.py) que permite navegar por la documentación técnica.

Características principales:

- Tablas ASCII: Formateo dinámico de tablas para una mejor visualización en consola.
- Análisis Estadístico: Integración con Pandas para mostrar promedios e importes en tiempo real.
- Robustez: Manejo de entradas inválidas para evitar cierres inesperados.

```bash
# Ejecución
python programa.py
```

## 📈 Análisis en Power BI

El reporte de Power BI (.pbix) incluye:

- Análisis de Outliers: Gráficos de dispersión para detectar transacciones que se desvían de la línea de regresión (precios erróneos o ventas premium).
- Evolución de Medios de Pago: Gráfico de áreas apiladas 100% para monitorear el uso de Efectivo, Tarjeta y QR.
- Time Intelligence: Análisis de ventas comparando periodos actuales frente a metas fijas.

## 🛠️ Tecnologías Utilizadas

- Lenguajes: Python (Pandas, Matplotlib, Seaborn).
- BI: Microsoft Power BI (DAX, Power Query).
- Base de Datos: Excel / CSV Relacional.
- Documentación: Markdown.
