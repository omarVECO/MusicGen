import torch
import torchaudio
import numpy as np
import torch.nn as nn
import wget
import os
import librosa
import csv
import sys
from utils.process import load_labels, preprocess_audio, mostrar_resultados_filtrados
sys.path.append('./audioset_tagging_cnn/pytorch')

# ======= CONFIGURACIÓN =======
MODEL_URL = 'https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1'
CHECKPOINT_PATH = 'Cnn14_16k.pth'  # Usa el nombre original
LABELS_CSV = 'class_labels_indices.csv'
AUDIO_PATH = 'test_audios/guitar_audio.wav'  # Cambia esto por tu archivo .wav
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ======= MODELO CNN14 =======
class Cnn14(torch.nn.Module):
    def __init__(self, sample_rate=16000, window_size=512, hop_size=160,
                 mel_bins=64, fmin=50, fmax=8000, classes_num=527):
        super().__init__()
        import sys
        sys.path.append('.')  # Asegura que podemos importar localmente
        from audioset_tagging_cnn.pytorch.models import Cnn14  # Asume que copiaste el modelo desde el repo oficial
        self.model = Cnn14(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num
        )
        self.model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)['model'])
        self.model.to(DEVICE).eval()

    def forward(self, waveform):
        with torch.no_grad():
            output = self.model(waveform.to(DEVICE))
        return output['clipwise_output'][0], output['embedding'][0]
    
# ======= MAIN =======
if __name__ == '__main__':
    print('Preparando audio...')
    audio = preprocess_audio(AUDIO_PATH)
    print('Cargando modelo...')
    model = Cnn14()

    print('Realizando predicción...')
    scores, embedding = model(audio)
    mostrar_resultados_filtrados(scores, LABELS_CSV)
    
    labels = load_labels(LABELS_CSV)
    top_k = torch.topk(scores, k=5)
    print('\nInstrumentos detectados:')
    for i in range(5):
        print(f'  - {labels[top_k.indices[i]]}: {top_k.values[i].item():.3f}')

    print(f'\nEmbedding shape: {embedding.shape}')
    np.save("embedding.npy", embedding.cpu().numpy())
    print('Embedding guardado como embedding.npy')