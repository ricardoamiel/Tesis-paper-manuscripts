import os
import cv2
import matplotlib.pyplot as plt

# --- Configuración de Tipografía Estilo LaTeX para la Tesis ---
plt.rcParams.update({
    "text.usetex": False,            # Cambiar a True si tienes LaTeX instalado en el sistema
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "font.size": 14,                 # Tamaño de fuente óptimo para títulos de figuras
    "axes.titlesize": 16,            # Tamaño de los encabezados (Image 1, etc.)
    "axes.labelsize": 12
})

# --- Configuración de Rutas de tu Dataset DSIFN ---
base_path = "Resultados_RALA-ChangeFormer/Datasets/DSIFN-CD-BENCH/train"  
dir_A = os.path.join(base_path, "A")
dir_B = os.path.join(base_path, "B")
dir_label = os.path.join(base_path, "label")

# --- Lectura Automatizada del Dataset (Sin carpeta 'list') ---
# Listamos los archivos .png dentro de la carpeta A para tomar el primero disponible
png_files = [f for f in os.listdir(dir_A) if f.lower().endswith('.png')]
if not png_files:
    raise FileNotFoundError(f"No se encontraron imágenes .png en la ruta: {dir_A}")

# Seleccionamos la primera imagen de la lista
img_name = png_files[0]

# Construir rutas completas de los archivos cruzando el mismo nombre
path_img1 = os.path.join(dir_A, img_name)
path_img2 = os.path.join(dir_B, img_name)
path_label = os.path.join(dir_label, img_name)

# Verificar que el archivo exista también en los directorios B y label antes de continuar
if not os.path.exists(path_img2):
    raise FileNotFoundError(f"La imagen {img_name} existe en 'A' pero falta en 'B': {path_img2}")
if not os.path.exists(path_label):
    raise FileNotFoundError(f"La imagen {img_name} existe en 'A' pero falta en 'label': {path_label}")

# --- Carga y Conversión de Imágenes ---
img1 = cv2.cvtColor(cv2.imread(path_img1), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread(path_img2), cv2.COLOR_BGR2RGB)
label = cv2.imread(path_label, cv2.IMREAD_GRAYSCALE)

# --- Generación de la Gráfica Triple ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')

# Columna 1: Image 1 (Tiempo A)
ax1.imshow(img1)
ax1.set_title("Image 1")
ax1.axis('off')  

# Columna 2: Image 2 (Tiempo B)
ax2.imshow(img2)
ax2.set_title("Image 2")
ax2.axis('off')

# Columna 3: Ground Truth (Máscara binaria de cambios)
ax3.imshow(label, cmap='gray')
ax3.set_title("Ground Truth")
ax3.axis('off')

# Optimizar la distribución espacial
plt.tight_layout()

# --- Guardar Resultado en PDF Vectorial ---
output_filename = 'DSIFN2.pdf'
plt.savefig(output_filename, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', dpi=300)

print(f"Visualización generada exitosamente y guardada como: {output_filename}")
print(f"Muestra procesada de forma emparejada: {img_name}")