"""
GESTION_DISPOSITIVOS.py - Gestion unificada del stream de audioasdas
"""

import sounddevice as sd
import numpy as np

class GestionDispositivos:
    def __init__(self, sample_rate=16000, blocksize=1024, channels=6):  # Cambiado a 1024
        self.sample_rate = sample_rate
        self.blocksize = blocksize  # Ahora 1024 para mejor rendimiento
        self.channels = 6
        self.device_index = None
        self.stream = None
        self.is_recording = False
        
        self.suscriptores = []
        
        self.encontrar_dispositivo_respeaker()
    
    def encontrar_dispositivo_respeaker(self):
        """Busca automaticamente el dispositivo Respeaker"""
        print("Buscando Respeaker 4 Mic Array...")
        try:
            dispositivos = sd.query_devices()
            for i, disp in enumerate(dispositivos):
                if "respeaker" in disp['name'].lower() and disp['max_input_channels'] >= 4:
                    self.device_index = i
                    print(f"Respeaker encontrado: [{i}] {disp['name']}")
                    return True
            
            print("Respeaker no encontrado. Dispositivos disponibles:")
            for i, disp in enumerate(dispositivos):
                if disp['max_input_channels'] >= 4:
                    print(f"   [{i}] {disp['name']} ({disp['max_input_channels']} canales)")
            return False
            
        except Exception as e:
            print(f"Error buscando dispositivo: {e}")
            return False
    
    def agregar_suscriptor(self, callback):
        """Agrega un modulo que recibira el audio"""
        self.suscriptores.append(callback)
        print(f"Suscriptor agregado. Total: {len(self.suscriptores)}")
    
    def audio_callback_central(self, indata, frames, time_info, status):
        """Unico callback que distribuye audio a todos los modulos"""
        if status:
            print(f"Status audio: {status}")
        
        if self.is_recording and indata is not None:
            for suscriptor in self.suscriptores:
                try:
                    suscriptor(indata.copy())
                except Exception as e:
                    print(f"Error en suscriptor: {e}")
    
    def iniciar_captura(self):
        """Inicia el unico stream de audio del sistema"""
        if self.device_index is None:
            print("Error: No hay dispositivo configurado")
            return False
        
        try:
            self.stream = sd.InputStream(
                device=self.device_index,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.blocksize,  # Usa 1024
                callback=self.audio_callback_central,
                dtype='float32'
            )
            
            self.is_recording = True
            self.stream.start()
            print(f"Captura de audio iniciada (Blocksize: {self.blocksize})")
            return True
            
        except Exception as e:
            print(f"Error iniciando captura: {e}")
            return False
    
    def detener_captura(self):
        """Detiene el stream de audio"""
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("Captura de audio detenida")