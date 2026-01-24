# OptiMaint – Sistema de Mantenimiento Predictivo Basado en Señales e Imágenes  
**Mondragon Unibertsitatea – Máster en Inteligencia Artificial Aplicada**  
**Autores:** Irene López, Julen Larrañaga, Jorge García, Izai Garay  
**Fecha:** Enero de 2026  

---

## Descripción general

**OptiMaint** es un sistema de **mantenimiento predictivo industrial** orientado a la **detección temprana de fallos**, combinando dos líneas principales:

- **Análisis de señales** para clasificación de fallos en rodamientos (*bearings*).
- **Visión artificial** para detección y localización de defectos visuales en piezas industriales.

El proyecto está organizado de forma **modular**, con scripts de entrenamiento/inferencia y **pipelines tipo productor–consumidor** (ejecución en procesos/terminales separados) para simular un flujo industrial.

---

## Estructura del proyecto

```text
OptiMaint/
├── bearings/                         # Modelado y análisis de fallos en rodamientos
│   ├── models/
│   │   ├── final_model/
│   │   │   └── final_model.pkl
│   │   ├── model_decision/
│   │   │   └── model_decision.py
│   │   ├── overhang/
│   │   │   └── overhang_model.pkl
│   │   └── underhang/
│   │       └── underhang_model.pkl
│   ├── optimization_model.py         # Optimización y selección de modelos
│   ├── overhang.py                   # Entrenamiento modelo overhang
│   ├── underhang.py                  # Entrenamiento modelo underhang
│   ├── Bearings_signal_analysis.ipynb# Análisis exploratorio de señales
│   ├── division_for_kafka.py          # Preparación de datos para el pipeline
│   └── division_over_under.py         # División jerárquica de fallos
│
├── bearings_pipeline/                # Pipeline productor–consumidor – rodamientos
│   ├── producer.py
│   ├── consumer.py
│   ├── consumer_overhang.py
│   ├── consumer_underhang.py
│   └── docker-compose.yml
│
├── defects/                          # Detección de defectos visuales en piezas
│   ├── models/
│   │   ├── best.pt                   # Modelo YOLO entrenado
│   │   └── visual_defect_detection.ipynb
│   └── defects_pipeline/
│       ├── producer.py
│       ├── consumer_yolo.py
│       ├── producer_consumer.py
│       └── docker-compose.yml
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Funcionamiento general

OptiMaint se divide en **dos bloques** que pueden ejecutarse por separado:

- **Bearings (señales):** clasifica el tipo de fallo a partir de características extraídas de señales.
- **Defects (imágenes):** detecta y localiza defectos visuales en piezas mediante YOLO.

Ambos se integran mediante **pipelines productor–consumidor**, pensados para ejecutarse en **varias terminales**.

---

## Bearings (señales)

### Qué hace
1. **Prepara datos** para el pipeline y divide el problema (general / overhang / underhang).
2. Entrena modelos (Random Forest) y guarda los modelos finales en `.pkl`.
3. En el pipeline:
   - un consumidor hace la **decisión general**
   - otros consumidores ejecutan la **clasificación específica** (overhang / underhang)

### Scripts principales
- `bearings/optimization_model.py`: selección/optimización del modelo.
- `bearings/overhang.py`, `bearings/underhang.py`: entrenamiento de submodelos.
- `bearings/models/*/*.pkl`: modelos finales serializados.

---

## Defects (imágenes)

### Qué hace
1. Ingiere imágenes de piezas.
2. Ejecuta un modelo YOLO (`best.pt`) para **detectar y localizar** defectos (bounding boxes).
3. Puede ejecutarse como flujo separado o en modo combinado productor+consumidor.

### Scripts principales
- `defects/models/visual_defect_detection.ipynb`: experimentación y preparación.
- `defects/models/best.pt`: modelo YOLO entrenado.
- `defects/defects_pipeline/consumer_yolo.py`: inferencia YOLO.

---

## Instalación

Desde la raíz del proyecto:

```bash
pip install -r requirements.txt
```

---

## Ejecución (en terminales separadas)

> **Importante:** cada script se ejecuta en una **terminal distinta**.  
> Abre todas las terminales en la **raíz del proyecto** (`OptiMaint/`).

### A) Bearings (señales)

Abre **4 terminales** y ejecuta:

**Terminal 1 – Producer (envía mensajes / datos al pipeline)**
```bash
python bearings_pipeline/producer.py
```
**Terminal 2 – Consumer general (clasificación principal)**
```bash
python bearings_pipeline/consumer.py
```

**Terminal 3 – Consumer overhang (clasificación específica)**
```bash
python bearings_pipeline/consumer_overhang.py
```

**Terminal 4 – Consumer underhang (clasificación específica)**
```bash
python bearings_pipeline/consumer_underhang.py
```
B) Defects (imágenes)

Abre 3 terminales y ejecuta:

**Terminal 1 – Producer (envía imágenes al pipeline)**
```bash
python defects/defects_pipeline/producer.py
```

**Terminal 2 – Producer-Consumer (flujo combinado auxiliar)**
```bash
python defects/defects_pipeline/producer_consumer.py
```

**Terminal 3 – Consumer (procesa mensajes del pipeline)**
```bash
python defects/defects_pipeline/consumer_yolo.py
```

---

## Resultados y almacenamiento (datalake)

Las predicciones generadas por los pipelines se **persisten en el datalake** para poder:
- auditar ejecuciones,
- analizar resultados a posteriori,
- y reutilizar salidas en procesos posteriores (reporting, visualización, etc.).

### Qué se guarda

- **Bearings (señales):**
  - Predicciones del **consumer general** (clasificación principal).
  - Predicciones de los **consumers específicos** (overhang / underhang).

- **Defects (imágenes):**
  - Predicciones de defecto/no defecto (si aplica en el flujo).
  - Resultados de YOLO (localización) asociados a cada imagen procesada.
  - (Opcional) referencias/rutas a outputs generados (por ejemplo, identificadores de imagen).

### Dónde se guardan

Las salidas se almacenan en la carpeta de datalake definida en el proyecto (por ejemplo, `datalake_bearings/` o el directorio configurado en cada pipeline).  
La estructura exacta depende de la configuración, pero la idea es mantener **un registro persistente** de las inferencias producidas por cada ejecución.

> Nota: el datalake se usa como almacenamiento de resultados y normalmente **no se versiona** en Git (ver `.gitignore`).
