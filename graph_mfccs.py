import librosa
import numpy as np
from sonopy import mfcc_spec, power_spec, mel_spec
import matplotlib.pyplot as plt
import soundfile as sf
from librosa.display import specshow
import argparse
    
def get_mfccs(audio_path,sample_rate, window_stride, fft_size, num_filt, num_coeffs, plot=1):
    # Load audio file
    audio, sr = sf.read(audio_path)
    #mfcc = sonopy.mfcc_spec(audio = audio, sample_rate=16000, window_stride=(400, 200), fft_size=400, num_filt=40, num_coeffs=40)
    powers, filters, mels, mfccs = mfcc_spec(audio = audio, 
                                             sample_rate=sample_rate, 
                                             window_stride=(window_stride*2, window_stride), 
                                             fft_size=fft_size, 
                                             num_filt=num_filt, 
                                             num_coeffs=num_coeffs, 
                                             return_parts=True)
    print("mfccs shape: ", mfccs.shape)

    if plot == 1:
        # Cargar el archivo de audio
        y, sr = librosa.load(audio_path)
        # Calcular el espectrograma de potencia
        D = np.abs(librosa.stft(y))
        # Convertir a decibeles
        DB = librosa.amplitude_to_db(D, ref=np.max)

        # Graficar señal de audio, espectrograma y MFCC
        plt.figure(figsize=(12, 6))
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(y=y, sr=sr, color="blue")
        plt.title('Señal de audio')
        plt.subplot(3, 1, 2)
        specshow(DB, sr=sr, x_axis='time', y_axis='hz')
        plt.title('Espectrograma')
        plt.subplot(3, 1, 3)
        specshow(mfccs.T, x_axis='time', sr=sr, hop_length=280, cmap='viridis')
        plt.yticks(ticks=range(0, 13, 3), labels=range(0, 13, 3))
        plt.title('MFCC')
        plt.xlabel('Time (s)')
        plt.ylabel('MFCC Coefficients')
        plt.tight_layout()
        plt.show()
    elif plot == 2:
        # Graficar
        plt.figure(figsize=(12, 6))
        specshow(DB, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Espectrograma')
        plt.tight_layout()
        plt.show()
    elif plot == 3:
        # Graficar los coeficientes MFCC
        plt.figure(figsize=(10, 4))
        #plt.imshow(mfccs.T, cmap='viridis', origin='lower', aspect='auto')
        specshow(mfccs.T, x_axis='time', sr=sr, hop_length=280, cmap='viridis')
        plt.yticks(ticks=range(0, 13, 2), labels=range(0, 13, 2))
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCC')
        plt.xlabel('Time (s)')
        plt.ylabel('MFCC Coefficients')
        plt.show()
    elif plot == 4:
        # Bancos de filtros
        plt.figure(figsize=(10, 4))
        plt.plot(filters.T)
        plt.title('Filtros de Mel')
        plt.xlabel('Frecuencia')
        plt.ylabel('Amplitud')
        plt.show()
    elif plot == 5:
        # Mel
        plt.figure(figsize=(10, 4))
        plt.plot(mels)
        plt.title('Mel ')
        plt.xlabel('Frecuencia')
        plt.ylabel('Amplitud')
        plt.show()
    elif plot == 6:
        plt.figure(figsize=(10, 4))
        specshow(powers.T, x_axis='time', sr=sr, hop_length=200, cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Power')
        plt.xlabel('Time (s)')
        plt.ylabel('Power')
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get MFCCs from audio file')
    parser.add_argument('--audio_path', type=str, required=True, help='Path to audio file')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate of audio file')
    parser.add_argument('--window_stride', type=int, default=200, help='Window stride in ms')
    parser.add_argument('--fft_size', type=int, default=512, help='FFT size')
    parser.add_argument('--num_filt', type=int, default=13, help='Number of filters')
    parser.add_argument('--num_coeffs', type=int, default=13, help='Number of coefficients')
    parser.add_argument('--plot', type=int, default=1, 
                        help='Plot MFCCs. 1. plot all, 2. plot spectrogram 3. plot mfccs, 4. plot filters, 5. plot mel, 6. plot power')

    args = parser.parse_args()
    get_mfccs(args.audio_path, args.sample_rate, args.window_stride, args.fft_size, args.num_filt, args.num_coeffs)