import torch
import torchaudio
import numpy as np
import torch.nn as nn
import wget
import os
import librosa
import csv
import sys
sys.path.append('./audioset_tagging_cnn/pytorch')

def load_labels(csv_path):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return [row['display_name'] for row in reader]

def preprocess_audio(wav_path, target_sample_rate=16000):
    waveform, sr = torchaudio.load(wav_path)
    waveform = waveform.mean(dim=0)  # convertir a mono
    waveform = torchaudio.functional.resample(waveform, sr, target_sample_rate)
    return waveform.unsqueeze(0)  # [1, T]

def mostrar_resultados_filtrados(scores, csv_path):
    instrumentos_validos = ["piano", "guitar", "bass"]
    
    # Cargar etiquetas
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        labels = [row["display_name"] for row in reader]
    
    resultados = []
    for i, score in enumerate(scores):
        nombre = labels[i].lower()
        if any(instr in nombre for instr in instrumentos_validos):
            resultados.append((labels[i], score.item()))

    # Ordenar por score descendente y mostrar top
    resultados.sort(key=lambda x: x[1], reverse=True)

    print("\n🎵 Instrumentos detectados (solo piano, guitarra y bajo):")
    if resultados:
        for label, score in resultados[:1]:
            print(f"  - {label}: {score:.3f}")
    else:
        print("  - Ninguno de los instrumentos especificados fue detectado.")