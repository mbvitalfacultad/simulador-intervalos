# Simulador de Intervalos de Confianza

Este proyecto es un **simulador interactivo** desarrollado en **Python + Streamlit** para trabajar con estudiantes avanzados de ingeniería en temas de **estadística inferencial**.  
El simulador permite visualizar y experimentar con **intervalos de confianza** en diferentes contextos:

- Intervalos para la **media con varianza conocida**  
- Intervalos para la **media con varianza desconocida**  
- Intervalos para la **varianza poblacional**  
- Intervalos para la **proporción poblacional**

---

## 🎯 Objetivos
- Favorecer la comprensión de los intervalos de confianza y su interpretación.  
- Brindar a los estudiantes la posibilidad de **simular múltiples muestras** y observar cómo los intervalos varían.  
- Comparar la cobertura de los intervalos con el **valor verdadero del parámetro poblacional**, representado con una **línea verde de referencia**.  
- Descargar los resultados en formato **CSV** para su análisis posterior.

---

### Requisitos
- Python 3.9 o superior  
- Librerías:  
  - `streamlit`  
  - `numpy`  
  - `pandas`  
  - `matplotlib`  
  - `scipy`  

📂 Descarga de simulaciones
Al finalizar una simulación, el usuario puede descargar los resultados en formato CSV con el botón correspondiente.
El archivo se guarda con el nombre fijo:

simulaciones.csv

📖 Uso en clase
El docente puede proponer experimentos cambiando parámetros como el tamaño muestral, el nivel de confianza o el número de simulaciones.

Los estudiantes pueden comparar visualmente la frecuencia con la que los intervalos incluyen el parámetro real.

Los archivos CSV permiten realizar un análisis adicional en Excel, R o Python.

✨ Créditos
Desarrollado con fines didácticos en el marco de la enseñanza de estadística en ingeniería.



