import os
import csv
import shutil
import sys

# --- CONFIGURACIÓN DE RUTAS ---
#CAMBIA ESTA RUTA
dataset_root = 'C:/Users/emili/Downloads/V2' 

csv_path = 'stems_metadata.csv'
output_dir = 'categorized_stems'
log_file_path = 'organization_log.txt'

# --- REDIRIGIR LA SALIDA A UN ARCHIVO DE TEXTO ---
original_stdout = sys.stdout
log_file = open(log_file_path, 'w', encoding='utf-8')
sys.stdout = log_file

# --- PROCESO PRINCIPAL ---
try:
    # Crear carpetas de destino
    categories = ['piano', 'guitar', 'bass', 'otros']
    for category in categories:
        path = os.path.join(output_dir, category)
        os.makedirs(path, exist_ok=True)
    print('Carpetas de destino creadas o ya existentes.')

    # Procesar CSV y mover archivos
    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)  # Saltar el encabezado

        for row in csv_reader:
            if len(row) < 2:
                continue

            stem_filename = row[0]
            label = row[1]
            
            # Obtenemos el nombre base de la canción
            song_name = stem_filename.split('_STEM_')[0]
            
            # Construimos el nombre exacto de la carpeta de stems para esta canción
            stems_folder_name = song_name + '_STEMS'
            
            # Construimos la ruta completa del archivo de origen
            source_path = os.path.join(dataset_root, song_name, stems_folder_name, stem_filename)
            
            # Construimos la ruta completa del destino
            destination_dir = os.path.join(output_dir, label)
            destination_path = os.path.join(destination_dir, stem_filename)

            if os.path.exists(source_path):
                # Usamos shutil.move para mover el archivo
                shutil.move(source_path, destination_path)
                # si lo quieres copiar en vez de que se mueva solo cambia el move por copy
                print(f"Movido: {stem_filename} -> {destination_dir}")
            else:
                print(f"Advertencia: Archivo no encontrado - {source_path}")

finally:
    # Restaurar la salida de la terminal y cerrar el archivo de log
    sys.stdout = original_stdout
    log_file.close()
    print(f"Proceso de organización completado. El registro se guardó en: {log_file_path}")