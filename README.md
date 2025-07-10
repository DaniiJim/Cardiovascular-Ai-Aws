# Cardiovascular-Ai-Aws


Proyecto desarrollado durante el programa AWS Educate por un grupo de estudiantes de informática. El objetivo es crear una aplicación web basada en inteligencia artificial para el análisis automático de electrocardiogramas (ECG), simulando los datos que capturan los smartwatches, con capacidad de diagnóstico e interpretación médica.

---

## 💡 Descripción del proyecto

El sistema tendrá dos componentes principales:

1. **Clasificación médica de ECGs**: un modelo de IA capaz de detectar patologías cardíacas (como fibrilación auricular) a partir de una única derivación del ECG, similar a la que ofrecen los smartwatches (por ejemplo, Apple Watch).

2. **Modelo generativo explicativo**: una IA explicativa (tipo LLM) que, en base al resultado del primer modelo, genere un texto comprensible para el usuario interpretando el diagnóstico.

---

## 🛠️ Tecnologías previstas

- **Lenguajes**: Python
- **Frameworks IA**: scikit-learn, PyTorch / TensorFlow (por decidir)
- **AWS Services** (según disponibilidad):
  - SageMaker (entrenamiento modelos)
  - S3 (almacenamiento de datos)
  - Lambda / API Gateway (backend)
  - Amplify / EC2 / LightSail (frontend web)

---

## 📊 Dataset

- Uso de datasets públicos de ECG multiderivación, como los disponibles en:
  - [PhysioNet](https://physionet.org/)
  - [CBIS-DDSM (para clasificación con imágenes)](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)
  - [PapersWithCode - ECG Classification](https://paperswithcode.com/datasets?task=ecg-classification)

Se priorizará el uso de datos que permitan extraer únicamente la derivación I, para simular lo que obtienen los dispositivos portátiles.

---

## 🌐 Funcionalidades previstas

- Interfaz web donde el usuario pueda:
  - Subir un archivo con su ECG (PDF o CSV).
  - Ver un diagnóstico automático.
  - Leer una interpretación generada en lenguaje natural por el modelo explicativo.

---

## 👥 Equipo

- Pilar  
- Jaime  
- Diego  
- Martin  
- Dani  

---

## 🔒 Licencia

Este proyecto está licenciado bajo la **GNU General Public License v3.0 (GPL-3.0)**.  
Puedes ver los términos completos aquí: [LICENSE](https://www.gnu.org/licenses/gpl-3.0.html)

---

## 🚧 Estado del proyecto

> **Fase actual:** recopilación de datasets, definición de arquitectura y primeros pasos con AWS.

---

## 💬 Contacto

Para dudas, sugerencias o colaboraciones, contacta con cualquier miembro del equipo o abre un issue en este repositorio.
