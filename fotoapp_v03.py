"""Módulo FotoApp: Herramientas de procesamiento y edición digital de imágenes.
Permite redimensionar fotos para redes sociales, ajustar contraste (Pillow y OpenCV),
aplicar filtros y detectar bordes/contornos mediante consola o importación.

Implementado durante el cursado de: Programación Python Avanzado - Tecnicatura Universitaria en Tecnologías de Programación (UPSO) - 
Bajo las cosignas del Proyecto Nº 2: "Procesamiento de imágenes y modelos de regresión con Python"
Segundo Cuatrimestre de 2026 - Fecha Entrega: 17 de Noviembre de 2026
"""

from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

import time
import requests


# Inicio de codigo de implementacion:
# A. Diccionarios y otros
# B. Bloque de funciones solicitadas
# C. main (para interactuar con usuario por consola) 


### como opcion, extra, se puede dar nombre a un imagen junto con la creacion de una carpeta
### en la implelentación se deja comentada, de manera que solo genere un archivo (al pasar por mostrar_comparar o cargar_imagen)
### dentro de la carpeta donde se este trabajando con las imagenes.

  
  # esto se usa (sacando las triples comillas) para guardar ls imagenes dentro de una carpeta
  # se debe retirar tambien las comillas dentro de las funciones: 1) cargar_imagen y 2) mostrar_comparar

Nueva_carpeta = "fotos_editadas"   # damos nombre a variable: "NuevaCarpeta"
    # Crea la carpeta si no existe; si ya existe de antes, no tira error (exist_ok=True)
os.makedirs(Nueva_carpeta, exist_ok=True)


#  ------ --- A

#### Definimos los (tres) diccionarios que usaremos desde distintas funciones

redes_sociales = {
    "1": {"nombre": "Instagram", "dimensiones": (1080, 1080)},
    "2": {"nombre": "Facebook", "dimensiones": (1200, 630)},
    "3": {"nombre": "Twitter", "dimensiones": (1200, 675)},
    "4": {"nombre": "YouTube", "dimensiones": (1280, 720)}
    }


filtros_pillow = {
    "1": {"nombre": "Desenfoque (BLUR)", "filtro": ImageFilter.BLUR, "codigo": "BLUR"},
    "2": {"nombre": "Contorno (CONTOUR)", "filtro": ImageFilter.CONTOUR, "codigo": "CONTOUR" },
    "3": {"nombre": "Detalle (DETAIL)", "filtro": ImageFilter.DETAIL, "codigo": "DETAIL"},
    "4": {"nombre": "Realce de bordes (EDGE_ENHANCE)", "filtro": ImageFilter.EDGE_ENHANCE, "codigo": "EDGE_ENHANCE"},
    "5": {"nombre": "Realce intenso (EDGE_ENHANCE_MORE)",
        "filtro": ImageFilter.EDGE_ENHANCE_MORE, "codigo": "EDGE_ENHANCE_MORE"},
    "6": {"nombre": "Relieve (EMBOSS)", "filtro": ImageFilter.EMBOSS, "codigo": "EMBOSS"},
    "7": {"nombre": "Detección de bordes (FIND_EDGES)", "filtro": ImageFilter.FIND_EDGES, "codigo": "FIND_EDGES"},
    "8": {"nombre": "Nitidez (SHARPEN)", "filtro": ImageFilter.SHARPEN, "codigo": "SHARPEN"},
    "9": {"nombre": "Suavizado (SMOOTH)", "filtro": ImageFilter.SMOOTH, "codigo": "SMOOTH"}
    }

tecnicas_contornos = {
    "1": {"nombre": "Bordes Canny", "codigo": "CANNY"},
    "2": {"nombre": "Umbralización Adaptativa", "codigo": "UMBRAL"},
    "3": {"nombre": "Filtro de Gradiente Sobel", "codigo": "SOBEL"}
    }

# ---  -- Termina código de los tres diccionarios implementados :)

# ---- B.
# Comienza las definicion de las dinstintas funciones implemantadas:

def cargar_imagen(direccion, aplicacion):
    
  """
  Esta función recibe como parámetros: 
  1) ruta para acceder a una imagen  y 
  2) clave numérica (como referencia a una red social).
  Luego de leer la imagen, la redimensiona a un tamaño dado asociado a la red social elegida (Ej: Instragram)
  utilizando thumbnail (Pillow) a fin de no distorsionar la proporción de la foto. 
  Finalmente guarda el archivo y devuelve la imagen procesada.
  """
 
  aplicacion = str(aplicacion).strip()  # se ingresará un nro (usuario). se convierte a 
                  # str para el diccionario y con .strip() se evitan espacios blanco involuntarios
  
  if aplicacion not in redes_sociales:
    raise ValueError(f"Opción '{aplicacion}' no válida. Elija un número del 1 al "
                  f"{len(redes_sociales)}.")

  direccion_limpia = (direccion).strip("\"' ")    # evita que la ruta/nombre imagen sea
                                              # cargada con "  o espacios bco inicio o fin
  imagen = Image.open(direccion_limpia).convert("RGB")
  tamano = redes_sociales[aplicacion]["dimensiones"]
  imagen.thumbnail(tamano)

  # guardado en "disco" y disponible en RAM (con el "return")

  nombre_red = redes_sociales[aplicacion]["nombre"]
  nombre_base = os.path.basename(direccion_limpia)   # basename toma el nombre del archivo original

  """
  # (*A) guardar archivo 
  nuevo_nombre_archivo = f"imagen_{nombre_red.lower()}_{nombre_base}" # con f-string anteponemos un str deseado
  imagen.save(nuevo_nombre_archivo)     # guardamos en "disco"

  print(
      f"Imagen redimensionada para: {nombre_red} {imagen.size} y guardada en/como:"
      f" {nuevo_nombre_archivo}")
  """

         # para usar, sacando comillas, para guardar en carpeta (NOTA: comentar lineas relac. a: (*A))
  nuevo_nombre_archivo = f"imagen_{nombre_red.lower()}_{nombre_base}" # con f-string anteponemos un str deseado
  ruta_guardado = os.path.join(Nueva_carpeta, nuevo_nombre_archivo)
  imagen.save(ruta_guardado)
  
  print(
      f"Imagen redimensionada para: {nombre_red} {imagen.size} y guardada en/como:"
      f" {ruta_guardado}")
    

  return imagen   # disponible en memoria RAM

# termina "cargar_imagen"

# -- --    -

def mostrar_comparar(img_original, img_procesada, titulo_derecha, nombre_archivo):

  """Esta Función muestra en conjunto la foto original y la foto editada en dos columnas.
  Para poder comparar los cambios antes y después de aplicada una dada transformación sobre la imagen.
  Finalmente se guarda en 'disco'.
  
  Puede ser usarsa con PIL y OpenCV.  e imagenes en ByN 
  
  Esta función puede ser llamada por el usuario o invocada por una de las funciones que aplican
  transformaciones a la imagen.  """

  # Convertimos las imagenes ent/sal a una matriz NumPy que nos permite unificar Pillow y OpenCV
  orig_arr = np.asarray(img_original)
  proc_arr = np.asarray(img_procesada)

  plt.figure(figsize=(14, 7))

  # Imagen Original
  plt.subplot(1, 2, 1)
  plt.imshow(orig_arr, cmap="gray", vmin=0, vmax=255)
  plt.title("Original", fontweight="bold", size=16)
  plt.axis("off")

  # Imagen Transformada
  plt.subplot(1, 2, 2)
  plt.imshow(proc_arr, cmap="gray", vmin=0, vmax=255)
  plt.title(titulo_derecha, fontweight="bold", size=16)
  plt.axis("off")

  plt.subplots_adjust(wspace=0.025)
  # Guardado portable (se hace siempre ANTES de plt.show())

  """
   # (*A) guardar archivo 
  plt.savefig(nombre_archivo)
  plt.show()
  """
         
  # para usar, sacando triple comillas, para guardar en carpeta (NOTA: comentar lineas relac. a: (*A))
  ruta_con_carpeta = os.path.join(Nueva_carpeta, nombre_archivo)
  plt.savefig(ruta_con_carpeta)
  plt.show()
  
# termina "mostrar_comparar"

# -- -  -- 

### Funciones para ajuste de constrante:

# 1. Ajustar constrante mediante Pillow y libreria ImageOps
# 2. Ajustar constrante mediante OpenCV (cv2) / CLAHE
# 3. Ajustar constrante mediante Pillow y libreria ImageEnhance (factor: pasado por defecto) 

# -- 1

def ajustar_contraste_pil(ruta_imagen):
  """ 
  Esta función Ecualiza el histograma de luminosidad, 
  detecta si la images es es blanco y negro o imagen color
  invoca a una tercer funcion para mostrar imagen inicial y transformada
  Finalmente guarda la figura en disco. 
  """

  # Limpiamos posibles comillas en la ruta y abrimos la imagen
  ruta_limpia = ruta_imagen.strip("\"' ")
  imagen = Image.open(ruta_limpia)

  # Detecta si la imagen esta en escala de grises ('L'). En este caso se puede aplicar directamente
  if imagen.mode == "L":
    imagen_final = ImageOps.equalize(imagen)

  else:
    # En caso de foto color, convertimos para asegurarnos de trabajar con formato RGB. Luego,
    # se separa la luminosidad/ brillo de "color". Se ecualizo el btillo (es decir, Y)
    imagen = imagen.convert("RGB")
    y, cb, cr = imagen.convert("YCbCr").split()
    y_ecualizado = ImageOps.equalize(y)
    # Finalmente, "juntamos" todo nuevamente y volvemos a RGB
    imagen_final = Image.merge("YCbCr", (y_ecualizado, cb, cr)).convert("RGB")

  nombre_base = os.path.basename(ruta_limpia)   # basename toma el nombre del archivo
                    # original y con f-string anteponemos un str deseado
  nombre_final_salida = f"editada_contraste_{nombre_base}"

  # Invocamos a la función "mostrar_comparar"
  mostrar_comparar(imagen, imagen_final, "Imagen Contraste Ecualizado",
                   nombre_final_salida)


# --- 2

def ajustar_contraste_cv2(ruta_imagen):
  """
  Esta función Ecualiza el contraste de forma adaptativa (CLAHE) usando OpenCV,
    soporta imágenes en blanco y negro o a color, muestra juntas la imagen original y
    la transformada para ser comparadas (invoca para esto a otra función)
    guarda la figura con nombre único.
    """
    
  # Limpiamos comillas y leemos la imagen tal cual viene (gris o color)
  ruta_limpia = ruta_imagen.strip("\"' ")
  img = cv2.imread(ruta_limpia, cv2.IMREAD_UNCHANGED)  # lee la matriz sin cambiarla, 1, 3 o 4 canales

  # herramienta CLAHE (cuadrícula 8x8 y límite  ruido 2.0)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

  # Verificamos si es escala de grises (2 dimensiones) o color (3 dimensiones)
  if img.ndim == 2:
    img_original_mostrar = img
    img_final_mostrar = clahe.apply(img)
  else:
    # Tomamos los 3 canales BGR.  (solo 3 , sin el 4to en caso de un PNG con transparencia)
    img_bgr = img[:, :, :3]

    #  pasamos de BGR a RGB para  Matplotlib y usaurio
    img_original_mostrar = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # con 4 canales (transp PNG) da error

    # Pasamos a YCrCb para aislar la luz ( i.e., Y) y aplicamos CLAHE solo a Y
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)
    y_ecualizado = clahe.apply(y)

    # Reconstruimos en BGR y pasamos a RGB para graficar bien en mostrar_comparar
    img_final_bgr = cv2.cvtColor(
        cv2.merge((y_ecualizado, cr, cb)), cv2.COLOR_YCrCb2BGR) # de YCrCb a BGR (Cv2)
    img_final_mostrar = cv2.cvtColor(img_final_bgr, cv2.COLOR_BGR2RGB) # a RGB

  # Armamos el nombre salida e invocamos a la func. "mostrar_comparar"
  nombre_base = os.path.basename(ruta_limpia)
  nombre_salida = f"editada_contraste_cv2_{nombre_base}"

  mostrar_comparar(img_original_mostrar, img_final_mostrar, "Contraste con CLAHE (OpenCV)",
      nombre_salida)

#  ---  3

def contraste_pillow_factor(ruta_imagen, factor=1.5):

  """ Esta funcion mejora el contraste de un imagen usando el módulo ImageEnhance de Pillow.
  (Ajuste "manual"). Por defecto tiene como parámetro factor = 1.5 
  (Nota: recordar. factor = 1.0 (sin cambios), > 1.0 (aumenta contraste), < 1.0 (reduce).

  Luego de la tranformacion de la imagen, invoca a una tercer funcion para mostrar la 
  imagen inicial y transformada.
  Finalmente guarda la figura en disco
  """
 
  ruta_limpia = ruta_imagen.strip("\"' ")
  imagen_original = Image.open(ruta_limpia).convert("RGB")

  # factor = 1.0 (sin cambios), > 1.0 (aumenta contraste), < 1.0 (reduce)
  imagen_final = ImageEnhance.Contrast(imagen_original).enhance(factor)


  nombre_base = os.path.basename(ruta_limpia)
  nombre_salida = f"editada_PIL_Enhance_ftr_{factor}_{nombre_base}"

  mostrar_comparar(imagen_original, imagen_final, f"Contraste con PIL (ImageEnhance - Factor: {factor})",
      nombre_salida)


#  Filtros de pillow (Image Filter) -- inicia

def aplicar_filtros(ruta_imagen, filtro):

  """ 
  Esta función utiliza el modulo de ImageFilter de Pillow.
  La func. recibe dos parámetros: 1) la ruta (nombre y extension de la imagen) y 2) el tipo de 
  filtro a aplicar (clave numérica) que se corresponde con la clave de un diccionario, el cual 
  contiene distintos filtros implementados (nueve).
  """
  
  filtro = str(filtro).strip()  # se ingresará un nro (usuario). se convierte a 
            # str para la clave del diccionario. strip() evita espacion bco. o saltos línea
  if filtro not in filtros_pillow:
    raise ValueError(f"Opción '{filtro}' no válida. Elija un número del 1 al"
        f" {len(filtros_pillow)}.")

  ruta_limpia = ruta_imagen.strip("\"' ")
  imagen_original = Image.open(ruta_limpia).convert("RGB")
  imagen_con_filtro = imagen_original.filter(filtros_pillow[filtro]["filtro"])
  
  nombre_base = os.path.basename(ruta_limpia)
  codigo_filtro_usado = (f"{filtros_pillow[filtro]['codigo']}")
  nombre_salida = f"editada_ImageFilter_{codigo_filtro_usado}_{nombre_base}"

  mostrar_comparar(imagen_original, imagen_con_filtro, 
                   f"ImageFilter - Filtro: {filtro}. {filtros_pillow[filtro]['nombre']}", nombre_salida)
  

# termina  "aplicar_filtro".

# --- 

# Función para detección de bordes 

def detectar_bordes_opciones(ruta_imagen, tecnica="1"):
  
  """
  Esta función permite resaltar bordes y contornos usando OpenCV.
  Las tecnicas implementadas son: 
  1. Canny, 
  2. Umbralización Adaptativa o 
  3. Sobel. 

  Luego de procesar la imagen (por una de las opciones elegidas), se muestra en conjunto 
  la figura original y la imagen procesada. 
  NOTA: la función muestra, para cada una de las 3 técnicas, la transformación por defecto. Es decir,
  contorno blanco y fondo negro (o viceserva en el caso de Umbraliz. Adaptiva.). En caso, de querer invertir
  la forma de la transformación (es decri, fondo blanco en lugar de negro, vicersa con bordes) se debe 
  descomentar las líneas con (*B). En este caso, permite guardar en disco y mostrar a partir de la func. externa
  "mostrar_comparar" tanto las transformaciones por defecto mencionadas como así también su "inversa" (i.e., ByN junto a NyB)
  """
  
  tecnica = str(tecnica).strip()
  if tecnica not in tecnicas_contornos:
    raise ValueError(f"Opción '{tecnica}' no válida. Elija del 1 al"
                      f" {len(tecnicas_contornos)}.")


  ruta_limpia = ruta_imagen.strip("\"' ")
  imagen_cont = cv2.imread(ruta_limpia)         # la lectura de OpenCv es: BGR

  if imagen_cont is None:       # si la ruta no esta bien escrita, cv2 no se rompería, devolvería un valor vacío
    print(f"La imagen no pudo ser leída. Por favor, revise la ubicación/nombre" 
          f"archivo de la imagen ingresada")
    return
  
  imagen_original_rgb = cv2.cvtColor(imagen_cont, cv2.COLOR_BGR2RGB)  # de BGR a RGB (para mostrar luego)
  imagen_cont_gris = cv2.cvtColor(imagen_cont, cv2.COLOR_BGR2GRAY) #  a escala de grises (GRAY)
    
  img_suavizada = cv2.GaussianBlur(imagen_cont_gris, (5, 5), 0)    # Suavizamos levemente para evitar 
                                              #  ruido se confunda con bordes reales (kernel de 5x5)
  
  # --
  # evalua con condicionales la opcion (tecnica) elegida por usaurio:

  if tecnica == "1":
    # Inicia opcion: Canny.
    img_bordes = cv2.Canny(img_suavizada, threshold1=80, threshold2=160) # (80, 160) son los umbrales 
                                      # mínimo y máximo de sensibilidad ((en gral proporc.: 1:2))
    #(*B) - ver docstring:
    #img_bordes_fdo_bco = cv2.bitwise_not(img_bordes) # para mostrar bordes en negro y fondo en blanco


  elif tecnica == "2":    # inicia tecnica Umbral adaptativo
    
    img_bordes = cv2.adaptiveThreshold(img_suavizada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)       # # 11: blockSize

    # Alternativas: ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV:
    #(*B) - ver docstring:
    """img_bordes_fdo_bco = cv2.adaptiveThreshold(img_suavizada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2) # fondo en negro y borde en blanco"""

  else:       # opción: Sobel (opcion 3)

    sobel_x = cv2.Sobel(img_suavizada, cv2.CV_64F, 1, 0, ksize=3)   # por eje x kernel de 3
    sobel_y = cv2.Sobel(img_suavizada, cv2.CV_64F, 0, 1, ksize=3)   # por eje y
    img_bordes = cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y)) # magnitude: para hipotenusa (luego valor abs.)

    #(*B) - ver docstring:
    #img_bordes_fdo_bco = 255 - img_bordes         # invertimos bco con negro (0 y 255 , respectivamente)


  # ---- damos nombre a la nueva imagen (imagen contorno)
  nombre_base = os.path.basename(ruta_limpia)


  codigo_tecnica = tecnicas_contornos[tecnica]["codigo"]
  nombre_tecnica = tecnicas_contornos[tecnica]["nombre"]
  nombre_salida = f"contorno_{codigo_tecnica}_{nombre_base}"
  nombre_salida2 = f"contorno_Inv_BN_{codigo_tecnica}_{nombre_base}"

  mostrar_comparar(imagen_original_rgb, img_bordes, f"Bordes: {nombre_tecnica}", nombre_salida)
  #(*B) - ver docstring:
  #mostrar_comparar(imagen_original_rgb, img_bordes_fdo_bco, f"Bordes: {nombre_tecnica}", nombre_salida2)

# termina deteccion de bordes ("standar") -- - -

# ----- 
#  NOTA !! La funcion a continuación, se implementa para canny con valores umbrales bajos (es decir, con
   # "detección sensible de contornos" )
   # "Canny de alta sensibilidad"

def detectar_bordes_sensible(ruta_imagen):
  
  """
  Función: "Canny de alta sensibilidad"
    Esta función permite resaltar bordes y contornos usando Canny. En este caso, los parámetros de 
    "threshold" son relativamente bajos. Esto permite contar con un detector de bordes (implementado en Canny) 
    mucho más sensible que los valores estandar habitualmente utilizados (100, 200 o 80, 160).
    Parametros Canny seteados: threshold1=20, threshold2=50.

    Esta implementació surgió, y se recomienda su uso para imagenes / fotos con cierta baja nitidez y 
    con bordes poco claros.
    En su concepción surgío al probar (Canny) con algunas fotos y, en particular, con una 
    foto: "pez payaso entre tentaculos de anémona".

    Finalmente, Luego del procesamiento de la imagen, se muestra la figura original junto a  la imagen procesada.   
    """

  ruta_limpia = ruta_imagen.strip("\"' ")
  imagen_cont = cv2.imread(ruta_limpia)         # la lectura de OpenCv es: BGR
  if imagen_cont is None:       # si la ruta no esta bien escrita, cv2 no se rompería, devolvería un valor vacío
    print(f"La imagen no pudo ser leída. Por favor, revise la ubicación/nombre" 
          f"archivo de la imagen ingresada")
    return
  
  imagen_original_rgb = cv2.cvtColor(imagen_cont, cv2.COLOR_BGR2RGB)  # de BGR a RGB (para mostrar luego)
  imagen_cont_gris = cv2.cvtColor(imagen_cont, cv2.COLOR_BGR2GRAY) #  a escala de grises (GRAY)
    
  img_suavizada = cv2.GaussianBlur(imagen_cont_gris, (5, 5), 0)    # Suavizamos levemente para evitar 
                                              #  ruido se confunda con bordes reales (kernel de 5x5)
  

  # Inicia opcion: Canny.
  img_bordes = cv2.Canny(img_suavizada, threshold1=20, threshold2=50) # () son los umbrales 
                    # mínimo y máximo de sensibilidad ((en gral proporc. 1: 2.5))
                    # teoria recomiendo en prop 1:2 a 1:3

  img_bordes_fdo_bco = cv2.bitwise_not(img_bordes) # para mostrar bordes en negro y fondo en blanco
  #img_bordes__fdo_bco2 = 255 - img_bordes         # invertimos bco con negro (0 y 255 , respectivamente)
  # NOTA: por resultado final pruebas se prefiere fondo negro y contorno blanco

  nombre_base = os.path.basename(ruta_limpia)
  nombre_salida = f"canny_sensible_{nombre_base}"
  nombre_salida2 = f"editada_Bordes_canny_fondo_bco_{nombre_base}"
  
  mostrar_comparar(imagen_original_rgb, img_bordes, f"Bordes: Canny (Sensible) ", nombre_salida)
  mostrar_comparar(imagen_original_rgb, img_bordes_fdo_bco, f"Bordes: Canny (Sensible)(Fdo. Blanco) ", nombre_salida2)

    # termina  "Canny de alta sensibilidad"

# termina bloque con funciones y tecnicas de detección de bordes/contornos

# -- --  - - 

# se creo una función que auxiliar independiente para una posible descarga de una imagen dede la web

def descargar_imagen_url(url, nombre_destino="imagen_web.jpg"):
  """ Esta función auxiliar (e independiente) permite descargar una imagen desde internet pasando dos
  parámetros: 1) la dirección web de la imagen (url) y 
  2) el nombre que recibirá la imagen al ser guardada en disco.
  La función utiliza la librería requests, estandar de python, para realizar peticiones web (http/https)
  """

  url_limpia = url.strip("\"' ")
  cabeceras = {"User-Agent": "Mozilla/5.0"}     # Se presenta como "un navegador web", 
                                                # para evitar bloqueo de servidores
                                          # que lo puede asociarlos con descargas automáticas
  respuesta = requests.get(url_limpia, headers=cabeceras, timeout=10) # hasta 10 seg para la respuesta de la página
  respuesta.raise_for_status()      # revisa la respuesta del servidor (200 OK, 403 Forbidden etc)
  with open(nombre_destino, "wb") as archivo: 
    archivo.write(respuesta.content)      # wb: write binary, es obligatorio para imágenes ya que no son 
                          # texto plano, vuelca adentro los bytes puros de la foto (respuesta.content)
                              # with: para que el archivo se cierre correctamente al terminar
  print(f"Imagen descargada correctamente como: {nombre_destino}")
  return nombre_destino


# --- - -   -- -

# A continuación DEfinimos la función "main" que será la intercción en el usuario

def main():

  """Esta función actua como menú interactivo por consola para utilizar las 
  herramientas desarrolladas de fotoApp"""

  print("   ======================  FOTOAPP  ================================\n")
  print("   ============      ¡BIENVENIDO A FOTOAPP!     ====================\n")
  print("   =====================  MENU CARGA  ==============================")

  
  # se ingresa com input a la ubicación de la imagen a procesar/transformar, previa
  # verificación de una ubicación/dirección valida de la imagen 
  
  while True:
    ruta_actual = input("\n Ingresá la ruta o nombre de la imagen inicial: ").strip("\"' ")
    if os.path.exists(ruta_actual):
      break
    print(f"\n Error: No se encontró el archivo '{ruta_actual}'. Verifique el nombre o "
          f"la ruta e intente nuevamente.")


  # Mostramos al usuario las distintas opciones para el procesamiento de las imagenes:

  while True:
      print(f"\n La imagen cargada es: 'Imagen actual: {ruta_actual}' \n")
      print(" · MENÚ DE OPCIONES:")
      print("1. Redimensionar Imagen para Redes Sociales")
      print("2. Ajustar Contraste (Ecualización Pillow o OpenCV)")
      print("3. Aplicar Filtros (Pillow)")
      print("4. Detectar Bordes y Contornos (OpenCV)")
      print("5. Cambiar de imagen (Nuevo archivo)")
      print("0. Salir")

      time.sleep(0.3)  # para dar tiempo a la interfaz gráfica (Colab) para renderizar 
                      # la imagen en pantalla (antes de un cuadro de texto en este caso)
                    
      opcion = input("\n Elegí una opción: ").strip()

      try:
        if opcion == "1":
          print("\n Redes sociales disponibles:")
          for clave, datos in redes_sociales.items():
            print(f"  {clave}. {datos['nombre']} {datos['dimensiones']}")
          red = input("Elija el número (tipo) de red social: ")
          cargar_imagen(ruta_actual, red)

        elif opcion == "2":
          print("\n Métodos de contraste disponibles:")
          print("  1. Ecualización Global (Pillow)")
          print("  2. Ecualización Adaptativa CLAHE (OpenCV)")
          print("  3. Ajuste Manual por Factor (Pillow) - Recomendado")

          sub_contraste = input("Elegí el método de contraste: ").strip()

          if sub_contraste == "1":
            ajustar_contraste_pil(ruta_actual)
          elif sub_contraste == "2":
            ajustar_contraste_cv2(ruta_actual)
          elif sub_contraste == "3":
            factor_ingresado = float(
                input("Ingrese el factor de contraste (ej. 1.5): ").strip())
            contraste_pillow_factor(ruta_actual, factor=factor_ingresado)
          else:
            print("Opción de contraste no válida.")

        elif opcion == "3":
          print("\n Filtros disponibles:")
          for clave, datos in filtros_pillow.items():
            print(f"  {clave}. {datos['nombre']}")
          filtro = input("Elija el número asociado al tipo de filtro: ")
          aplicar_filtros(ruta_actual, filtro)

        elif opcion == "4":
          print("\n Técnicas de contornos disponibles:")
          for clave, datos in tecnicas_contornos.items():
            print(f"  {clave}. {datos['nombre']}")
          tecnica = input("Elegí la técnica: ")
          detectar_bordes_opciones(ruta_actual, tecnica)

        elif opcion == "5":
            while True:
              ruta_actual = input("\n Ingrese la ruta o nombre de la nueva imagen: ").strip("\"' ")
              if os.path.exists(ruta_actual):
                break
              print(f"\n Error: No se encontró el archivo '{ruta_actual}'. Verifique el nombre o "
                    f"la ruta e intente nuevamente.")

        elif opcion == "0":
          print("\n Gracias por usar la Aplicación: FotoApp! Programa finalizado.")
          break

        else:
          print("\n Opción no válida. Por favor, elegí un número del 0 al 5.")

      except ValueError as e:
        print(f"\n Error en la opción ingresada: {e}")
      except FileNotFoundError:
        print("\n No se encontró la imagen en esa ruta ingresada. Por favor, use la opción 5 para"
            " cargar una ruta válida.")
      except Exception as e:
        print(f"\n Ocurrió un error inesperado: {e}")


if __name__ == "__main__":
  main()





