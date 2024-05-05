"""the interface to interact with wakeword model"""
import pyaudio
import threading
import time
import argparse
import wave
import torchaudio
import torch
import numpy as np
from neuralnet.dataset import get_featurizer
from threading import Event
import soundfile as sf
from scipy.signal import lfilter
import os 

class Listener:

    def __init__(self, sample_rate=16000, record_seconds=2):
        self.chunk = 1024
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        output=True,
                        frames_per_buffer=self.chunk)
        self.paused = Event()  # Evento para pausar el hilo
        self.paused.clear() # Inicialmente la escucha está pausada

    def __end__(self):
        self.stream.stop_stream() # Detiene la captura de audio
        self.stream.close() # Cierra el stream de audio
        self.p.terminate() # Termina la instancia PyAudio
        
    def listen(self, queue):
        while True:
            if not self.paused.is_set():  # Verifica si la escucha está pausada
                data = self.stream.read(self.chunk , exception_on_overflow=False)
                queue.append(data)
            time.sleep(0.01)

    def run(self, queue):
        thread = threading.Thread(target=self.listen, args=(queue,), daemon=True)
        thread.start()
        print("\nWake Word Engine is now listening... \n")


class WakeWordEngine:

    def __init__(self, model_file, filter_path):
        self.listener = Listener(sample_rate=16000, record_seconds=2)
        self.model = torch.jit.load(model_file)
        self.model.eval().to('cpu')  #run on cpu
        self.featurizer = get_featurizer(sample_rate=16000)
        self.audio_q = list()
        self.win = np.load(os.path.join(filter_path, 'CoeficientesFPB.npy'))
        
    def save(self, waveforms, fname="wakeword_temp"):
        wf = wave.open(fname, "wb")
        # set the channels
        wf.setnchannels(1)
        # set the sample format
        wf.setsampwidth(self.listener.p.get_sample_size(pyaudio.paInt16))
        # set the sample rate
        wf.setframerate(16000)
        # write the frames as bytes
        wf.writeframes(b"".join(waveforms))
        # close the file
        wf.close()
        audio, sr = sf.read(fname)
        
        return fname, audio
    
    def filer_audio(self, audio, sample_rate=16000, fname="wakeword_temp"):
        # Aplicar filtro
        AudioFiltrado = lfilter(self.win, 1, audio)
        sf.write(fname,AudioFiltrado,sample_rate,format='wav')
        return fname

    def predict(self, audio):
        with torch.no_grad():
            fname, audio = self.save(audio)
            # Aplicar filtro
            fname = self.filer_audio(audio=audio, fname=fname)
            
            waveform, _ = torchaudio.load(fname, normalize=False)  # don't normalize on train
            mfcc = self.featurizer(waveform).transpose(1, 2).transpose(0, 1)

            # TODO: read from buffer instead of saving and loading file
            # waveform = torch.Tensor([np.frombuffer(a, dtype=np.int16) for a in audio]).flatten()
            # mfcc = self.featurizer(waveform).transpose(0, 1).unsqueeze(1)

            out = self.model(mfcc)
            pred = torch.round(torch.sigmoid(out))
            return pred.item()

    def inference_loop(self, action):
        while True:
            if len(self.audio_q) > 15:  # remove part of stream
                diff = len(self.audio_q) - 15
                for _ in range(diff):
                    self.audio_q.pop(0)
                action(self.predict(self.audio_q))
            elif len(self.audio_q) == 15:
                action(self.predict(self.audio_q))
            time.sleep(0.05)

    def run(self, action):
        #self.wake_word_engine.resume_listening()  # Reanuda la escucha
        self.listener.run(self.audio_q)
        thread = threading.Thread(target=self.inference_loop,
                                    args=(action,), daemon=True)
        thread.start()
    
    def pause_listening(self):
        print('Antes de pausar:', self.listener.paused.is_set()) # Muestra el estado de la escucha
        self.listener.paused.set()  # Pausa la escucha
        print("Escucha pausada", self.listener.paused.is_set())

    def resume_listening(self):
        print("Antes de reanudar", self.listener.paused.is_set()) # Muestra el estado de la escucha
        # Restablecer contadores o flags de estado
        self.detect_in_row = 0
        
        # Limpiar archivos temporales
        if os.path.exists("wakeword_temp.wav"):
            os.remove("wakeword_temp.wav")
        
        # Vaciar la cola de audio
        self.audio_q.clear()
        
        # Reanudar la escucha
        self.listener.paused.clear()  # Reanuda la escucha

        print("Escucha reanudada", self.listener.paused.is_set())
        


class DemoAction:
    """
        args: sensitivty. the lower the number the more sensitive the
        wakeword is to activation.
    """
    def __init__(self, sensitivity=10, wake_word_engine=None):
        self.detect_in_row = 0
        self.sensitivity = sensitivity
        self.wake_word_engine = wake_word_engine

    def __call__(self, prediction):
        if prediction == 1:
            self.detect_in_row += 1
            print(self.detect_in_row)
            if self.detect_in_row >= self.sensitivity:
                # Llama a la funcion de acción
                self.wake_word_engine.pause_listening()
                lunaactivado(self.wake_word_engine)
                self.detect_in_row = 0
        else:
            self.detect_in_row = 0
            print('.')

def lunaactivado(wake_word_engine):
    print("Luna activado")
    time.sleep(2)
    wake_word_engine.resume_listening()  # Reanuda la escucha
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demoing the wakeword engine")
    parser.add_argument('--model_file', type=str, default=None, required=True,
                        help='optimized file to load. use optimize_graph.py')
    parser.add_argument('--sensitivity', type=int, default=10, required=False,
                        help='lower value is more sensitive to activations')
    parser.add_argument('--filter_path', type=str, default=None, required=True,
                        help='path to the filter coefficients')

    args = parser.parse_args()
    wakeword_engine = WakeWordEngine(args.model_file, args.filter_path)
    action = DemoAction(args.sensitivity, wake_word_engine=wakeword_engine)

    # action = lambda x: print(x)
    wakeword_engine.run(action)
    threading.Event().wait()
