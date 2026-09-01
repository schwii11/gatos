# tarea-02

## Integrantes

- Integrante 1: **Isidora Perez**
- Integrante 2: **Katalina Rios**

## Asignatura

**Dispositivos Periféricos y Plataformas para la Interacción Digital — DIS9087**

## Descripción del proyecto

Proyecto de reconocimiento de gestos utilizando Python, MediaPipe y una cámara web. La aplicación identifica gestos de las manos y expresiones faciales, y muestra en pantalla una imagen de hámster asociada a cada interacción.

El proyecto fue realizado tomando como referencia el repositorio:

https://github.com/catherpiee/meowmeowcatcam

## Cambios realizados

- Se reemplazaron los memes de gatos originales por nueve imágenes nuevas de hámsters.
- Se eliminaron las imágenes antiguas de gatos de la carpeta `memes/`.
- Se modificó `app.js`, correspondiente a la versión web de la aplicación.
- Se modificó `gesture_meme.py`, correspondiente a la versión de escritorio en Python.
- Se añadió reconocimiento de sonrisa mediante los *blendshapes* faciales de MediaPipe.
- Se actualizaron los gestos para que sean distintos de los utilizados en el proyecto de referencia.
- Se actualizó `index.html` para mostrar el hámster predeterminado al iniciar la aplicación.

## Tecnologías utilizadas

- Python
- MediaPipe
- OpenCV
- JavaScript
- HTML y CSS
- Cámara web

## Gestos

| # | Nombre | Cómo se activa | Imagen |
|---|---|---|---|
| 1 | Silly hamster | No realizar ningún gesto. | `sillyhamster.jpeg` |
| 2 | Corazón hamster | Hacer un corazón coreano con una mano, juntando pulgar e índice. | `corazonhamster.jpeg` |
| 3 | Feliz hamster | Sonreír y abrir la boca. | `felizhamster.jpeg` |
| 4 | Gym hamster | Levantar un puño cerca de la parte superior de la cara. | `gymhamster.jpeg` |
| 5 | Hamster | Juntar ambas palmas abiertas. | `hamster.jpeg` |
| 6 | Muehjej hamster | Unir las puntas de ambos índices y también las puntas de ambos pulgares. | `muehjejhamster.jpeg` |
| 7 | No sé hamster | Levantar ambas palmas abiertas hacia arriba y hacia los lados, como un gesto de pregunta. | `nosehamster.jpeg` |
| 8 | Paz hamster | Levantar el índice y el dedo medio de una mano, manteniendo los demás cerrados. | `pazhamster.jpeg` |
| 9 | Pizza hamster | Mantener ambas manos abiertas apuntándose entre sí. | `pizzahamster.jpeg` |

## Ajustes de detección

- **Corazón hamster:** utiliza un corazón coreano de una mano, acercando el pulgar y el índice.
- **Muehjej hamster:** requiere que se toquen las puntas de ambos índices y también las puntas de ambos pulgares.
- **Hamster:** se activa al juntar ambas palmas; se permite una pequeña separación para que la cámara detecte las dos manos.
- **No sé hamster:** se activa con ambas manos abiertas y separadas, en la postura de pregunta con las palmas hacia arriba y hacia los lados.

## Carpeta de imágenes

Todas las imágenes utilizadas por la aplicación se encuentran en:

```text
memes/
```

## Ejecución de la aplicación web

Desde la carpeta raíz del proyecto, ejecutar:

```powershell
python -m http.server 8001
```

Luego abrir en el navegador:

```text
http://localhost:8001
```

Se debe permitir el acceso a la cámara web. Para recargar la versión más reciente del proyecto en el navegador, utilizar `Ctrl + F5`.

## Ejecución con Python

Instalar las dependencias y ejecutar:

```powershell
python -m pip install -r requirements.txt
python gesture_meme.py
```

Presionar `q` o `Esc` para cerrar la aplicación.

## Video demostrativo

Agregar aquí el enlace o archivo de un video de máximo 30 segundos que muestre la aplicación en funcionamiento y al menos seis de los gestos implementados.

**Video:** Pendiente de agregar.
