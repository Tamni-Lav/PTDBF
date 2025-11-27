"""
GESTION_DISPOSITIVOS.py - Gestion unificada del stream de audio
vcaa - CON CORRECCIÓN PARA OVERFLOW
"""

import sounddevice as sd
import numpy as np
import threading
import time

class GestionDispositivos:
    def __init__(self, sample_rate=16000, blocksize=1024, channels=6):
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.channels = channels
        self.device_index = None
        self.stream = None
        self.is_recording = False
        
        # ✅ MEJORA: Control de overflow
        self.callback_lock = threading.Lock()
        self.suscriptores = []
        self.overflow_counter = 0
        self.last_overflow_time = 0
        self.processing_time = 0
        
        self.encontrar_dispositivo_respeaker()
    
    def encontrar_dispositivo_respeaker(self):
        """Busca automaticamente el dispositivo Respeaker"""
        print("Buscando Respeaker 4 Mic Array...")
        try:
            dispositivos = sd.query_devices()
            for i, disp in enumerate(dispositivos):
                if "respeaker" in disp['name'].lower() and disp['max_input_channels'] >= 4:
                    self.device_index = i
                    print(f"✅ Respeaker encontrado: [{i}] {disp['name']}")
                    return True
            
            print("⚠️ Respeaker no encontrado. Dispositivos disponibles:")
            for i, disp in enumerate(dispositivos):
                if disp['max_input_channels'] >= 4:
                    print(f"   [{i}] {disp['name']} ({disp['max_input_channels']} canales)")
            return False
            
        except Exception as e:
            print(f"❌ Error buscando dispositivo: {e}")
            return False
    
    def agregar_suscriptor(self, callback):
        """Agrega un modulo que recibira el audio"""
        with self.callback_lock:
            self.suscriptores.append(callback)
        print(f"✅ Suscriptor agregado. Total: {len(self.suscriptores)}")
    
    def remover_suscriptor(self, callback):
        """Remueve un suscriptor del sistema"""
        with self.callback_lock:
            if callback in self.suscriptores:
                self.suscriptores.remove(callback)
                print(f"✅ Suscriptor removido. Total: {len(self.suscriptores)}")
            else:
                print("⚠️ Suscriptor no encontrado")
    
    def audio_callback_central(self, indata, frames, time_info, status):
        """Unico callback que distribuye audio a todos los modulos"""
        start_time = time.time()
        
        if status:
            if status.input_overflow:
                self.overflow_counter += 1
                current_time = time.time()
                if current_time - self.last_overflow_time > 2.0:  # Mostrar cada 2 segundos máximo
                    print(f"⚠️ OVERFLOW #{self.overflow_counter}: El sistema no puede procesar tan rápido")
                    self.last_overflow_time = current_time
        
        if self.is_recording and indata is not None:
            # ✅ MEJORA: Procesamiento más eficiente
            with self.callback_lock:
                for suscriptor in self.suscriptores:
                    try:
                        suscriptor(indata.copy())
                    except Exception as e:
                        print(f"❌ Error en suscriptor de audio: {e}")
        
        # Medir tiempo de procesamiento
        self.processing_time = time.time() - start_time
        
        # Advertencia si el procesamiento es muy lento
        if self.processing_time > 0.1:  # Más de 100ms es problemático
            print(f"⚠️ Procesamiento lento: {self.processing_time:.3f}s")
    
    def obtener_estado(self):
        """Devuelve el estado actual del sistema"""
        return {
            'recording': self.is_recording,
            'device_index': self.device_index,
            'sample_rate': self.sample_rate,
            'blocksize': self.blocksize,
            'channels': self.channels,
            'suscriptores': len(self.suscriptores),
            'overflows': self.overflow_counter,
            'processing_time': self.processing_time
        }
    
    def listar_dispositivos(self):
        """Lista todos los dispositivos de audio disponibles"""
        try:
            dispositivos = sd.query_devices()
            print("\n" + "="*60)
            print("DISPOSITIVOS DE AUDIO DISPONIBLES")
            print("="*60)
            
            for i, disp in enumerate(dispositivos):
                tipo = "ENTRADA" if disp['max_input_channels'] > 0 else "SALIDA"
                canales = f"{disp['max_input_channels']}in/{disp['max_output_channels']}out"
                seleccionado = " ✅" if i == self.device_index else ""
                print(f"[{i:2d}]{seleccionado} {disp['name'][:40]:40} {tipo:8} {canales:10} {disp['default_samplerate']}Hz")
            
            print("="*60)
            return dispositivos
            
        except Exception as e:
            print(f"❌ Error listando dispositivos: {e}")
            return []
    
    def seleccionar_dispositivo_manual(self, device_index):
        """Permite seleccionar manualmente un dispositivo"""
        try:
            dispositivos = sd.query_devices()
            if 0 <= device_index < len(dispositivos):
                dispositivo = dispositivos[device_index]
                if dispositivo['max_input_channels'] >= 4:
                    self.device_index = device_index
                    print(f"✅ Dispositivo seleccionado: [{device_index}] {dispositivo['name']}")
                    return True
                else:
                    print(f"❌ Dispositivo no tiene suficientes canales de entrada (necesita 4, tiene {dispositivo['max_input_channels']})")
                    return False
            else:
                print(f"❌ Índice de dispositivo inválido: {device_index}")
                return False
                
        except Exception as e:
            print(f"❌ Error seleccionando dispositivo: {e}")
            return False
    
    def iniciar_captura(self):
        """Inicia el unico stream de audio del sistema"""
        if self.device_index is None:
            print("❌ Error: No hay dispositivo configurado")
            self.listar_dispositivos()
            return False
        
        if self.is_recording:
            print("⚠️ La captura de audio ya está activa")
            return True
        
        try:
            # ✅ MEJORA: Aumentar buffers para evitar overflow
            self.stream = sd.InputStream(
                device=self.device_index,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                callback=self.audio_callback_central,
                dtype='float32',
                latency='high',  # ✅ Más estabilidad
                extra_settings=None  # ✅ Sin configuraciones extra que puedan causar problemas
            )
            
            self.is_recording = True
            self.stream.start()
            
            print(f"✅ Captura de audio iniciada:")
            print(f"   - Dispositivo: {self.device_index}")
            print(f"   - Sample Rate: {self.sample_rate} Hz")
            print(f"   - Blocksize: {self.blocksize} muestras")
            print(f"   - Canales: {self.channels}")
            print(f"   - Suscriptores: {len(self.suscriptores)}")
            print(f"   - Modo: Alta latencia (más estable)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error iniciando captura: {e}")
            self.is_recording = False
            return False
    
    def pausar_captura(self):
        """Pausa la captura de audio temporalmente"""
        if self.stream and self.is_recording:
            self.stream.stop()
            self.is_recording = False
            print("⏸️ Captura de audio pausada")
            return True
        return False
    
    def reanudar_captura(self):
        """Reanuda la captura de audio"""
        if self.stream and not self.is_recording:
            self.stream.start()
            self.is_recording = True
            print("▶️ Captura de audio reanudada")
            return True
        return False
    
    def detener_captura(self):
        """Detiene completamente el stream de audio"""
        self.is_recording = False
        
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                self.stream = None
                print("🛑 Captura de audio detenida completamente")
                if self.overflow_counter > 0:
                    print(f"   - Overflows totales: {self.overflow_counter}")
            except Exception as e:
                print(f"⚠️ Error cerrando stream: {e}")
    
    def cambiar_blocksize(self, nuevo_blocksize):
        """Cambia el blocksize (requiere reiniciar captura)"""
        if self.is_recording:
            print("⚠️ No se puede cambiar blocksize durante la captura. Detenga primero.")
            return False
        
        if nuevo_blocksize not in [256, 512, 1024, 2048]:
            print("⚠️ Blocksize debe ser 256, 512, 1024 o 2048")
            return False
        
        self.blocksize = nuevo_blocksize
        print(f"✅ Blocksize cambiado a: {nuevo_blocksize}")
        return True
    
    def __del__(self):
        """Destructor para limpieza segura"""
        self.detener_captura()