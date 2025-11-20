"""
BEAMFORMING.py - Sistema de beamforming con espectrogramas comparativos
vcaa
"""

import numpy as np
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from scipy import signal
from scipy.io import wavfile
import os
import time

class BeamformingSystem:
    def __init__(self, gestion_audio, doa_system):
        self.gestion_audio = gestion_audio
        self.doa = doa_system
        
        # Configuracion de audio
        self.sample_rate = 16000
        self.blocksize = 1024  # Aumentado para mejor rendimiento
        
        # Configuracion del array
        self.radio = 0.0325
        self.mic_positions = np.array([
            [-self.radio, 0],    # Microfono 1 (canal 2)
            [0, -self.radio],    # Microfono 2 (canal 3)  
            [self.radio, 0],     # Microfono 3 (canal 4)
            [0, self.radio]      # Microfono 4 (canal 5)
        ])
        
        self.sound_speed = 343.0
        
        # Estado del sistema
        self.is_active = False
        self.is_processing = False
        self.current_angle = 0
        
        # Buffers para espectrogramas (5 segundos)
        self.buffer_duration = 5  # segundos
        self.buffer_size = self.buffer_duration * self.sample_rate
        self.audio_buffer = np.zeros((self.buffer_size, 4))
        self.beamformed_buffer = np.zeros(self.buffer_size)
        self.full_beamformed_audio = []  # Buffer completo para guardar
        
        # Configuracion de guardado
        self.output_folder = r"C:\Users\leona\Desktop\TLTech\PFJdN\Audios_Beamformed"
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Pre-calculos
        self.delays_precalculated = self._precalculate_all_delays()
        
        # Visualizacion
        self.fig = None
        self.ax_mic1 = None
        self.ax_beam = None
        self.ax_info = None
        self.mic1_spec_image = None
        self.beam_spec_image = None
        self.angle_text = None
        
        # Registrarse para recibir audio
        self.gestion_audio.agregar_suscriptor(self.recibir_audio)
        
        print("Beamforming inicializado")
        print(f" - Blocksize: {self.blocksize} muestras")
        print(f" - Duracion visualizacion: {self.buffer_duration} segundos")
        print(f" - Audio beamformed se guardara en: {self.output_folder}")

    def _precalculate_all_delays(self):
        """Pre-calcula todos los delays para 360 grados"""
        delays = np.zeros((360, 4))
        
        for angle_deg in range(360):
            angle_rad = np.deg2rad(angle_deg)
            direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            
            for mic in range(4):
                distance = np.dot(self.mic_positions[mic], direction)
                delays[angle_deg, mic] = -distance / self.sound_speed
            
            delays[angle_deg] -= np.min(delays[angle_deg])
            
        return delays

    def calculate_delays(self, angle_deg):
        """Obtiene delays pre-calculados para un angulo especifico"""
        angle_idx = int(round(angle_deg)) % 360
        return self.delays_precalculated[angle_idx].copy()

    def apply_beamforming(self, audio_data, angle_deg):
        """Aplica beamforming usando el metodo Sum and Delay"""
        delays = self.calculate_delays(angle_deg)
        
        delay_samples = np.round(delays * self.sample_rate).astype(int)
        max_delay = np.max(delay_samples)
        
        output_length = len(audio_data) + max_delay
        aligned_signals = np.zeros((output_length, 4))
        
        for mic in range(4):
            start_idx = delay_samples[mic]
            end_idx = start_idx + len(audio_data)
            if end_idx <= output_length:
                aligned_signals[start_idx:end_idx, mic] = audio_data[:, mic]
        
        beamformed_signal = np.sum(aligned_signals, axis=1) / 4.0
        return beamformed_signal[:len(audio_data)]

    def recibir_audio(self, audio_data):
        """Callback que recibe audio del gestor central"""
        if self.is_active and audio_data is not None and self.is_processing:
            try:
                if audio_data.shape[1] >= 5:
                    mic_data = audio_data[:, 1:5].copy()
                    
                    # Actualizar buffer para espectrogramas (5 segundos)
                    self.audio_buffer = np.roll(self.audio_buffer, -len(mic_data), axis=0)
                    self.audio_buffer[-len(mic_data):, :] = mic_data
                    
                    # Obtener angulo actual del DOA
                    self.current_angle = self.doa.angulo_actual
                    
                    # Aplicar beamforming
                    beamformed = self.apply_beamforming(mic_data, self.current_angle)
                    
                    # Actualizar buffer beamformed para visualizacion
                    self.beamformed_buffer = np.roll(self.beamformed_buffer, -len(beamformed))
                    self.beamformed_buffer[-len(beamformed):] = beamformed
                    
                    # Guardar en buffer completo para archivo WAV
                    self.full_beamformed_audio.append(beamformed.copy())
                        
            except Exception as e:
                print(f"Error en beamforming: {e}")

    def compute_spectrogram(self, data, nperseg=512):
        """Calcula el espectrograma de una señal"""
        f, t, Sxx = signal.spectrogram(data, self.sample_rate, nperseg=nperseg, noverlap=nperseg//2)
        return f, t, 10 * np.log10(Sxx + 1e-10)  # Convertir a dB

    def guardar_audio_beamformed(self):
        """Guarda el audio beamformed completo en archivo WAV"""
        if len(self.full_beamformed_audio) == 0:
            print("No hay datos de audio beamformed para guardar")
            return

        try:
            # Concatenar todos los bloques de audio
            audio_data = np.concatenate(self.full_beamformed_audio, axis=0)
            
            # Convertir a formato WAV (16-bit)
            audio_int16 = np.int16(audio_data * 32767)
            
            # Crear nombre de archivo con timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"beamformed_{timestamp}.wav"
            filepath = os.path.join(self.output_folder, filename)
            
            # Guardar archivo WAV
            wavfile.write(filepath, self.sample_rate, audio_int16)
            
            # Estadisticas
            duracion = len(audio_data) / self.sample_rate
            print(f"Audio beamformed guardado: {filepath}")
            print(f"Duracion: {duracion:.2f} segundos")
            print(f"Tamaño: {len(audio_data)} muestras")
            
            # Limpiar buffer despues de guardar
            self.full_beamformed_audio = []
            
        except Exception as e:
            print(f"Error guardando audio beamformed: {e}")

    def configurar_visualizacion(self, fig):
        """Configura la visualizacion con dos espectrogramas uno sobre otro"""
        try:
            self.fig = fig
            self.fig.clear()
            
            # Crear layout vertical: 2 espectrogramas + area de info
            gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 0.15])
            
            # Titulo principal
            self.fig.suptitle(f'BEAMFORMING - Comparacion de Espectrogramas ({self.buffer_duration}s)', 
                            fontsize=16, fontweight='bold')
            
            # Espectrograma del primer microfono (canal 2) - ARRIBA
            self.ax_mic1 = fig.add_subplot(gs[0])
            
            # Inicializar con datos vacios (5 segundos)
            empty_spec = np.zeros((100, 100))
            self.mic1_spec_image = self.ax_mic1.imshow(empty_spec, 
                                                      aspect='auto', cmap='viridis',
                                                      origin='lower', 
                                                      extent=[0, self.buffer_duration, 0, 4000])
            self.ax_mic1.set_title('Microfono 1 (Canal 2) - Señal Original', fontweight='bold')
            self.ax_mic1.set_ylabel('Frecuencia (Hz)')
            self.ax_mic1.set_ylim(0, 4000)
            self.ax_mic1.set_xlim(0, self.buffer_duration)
            self.ax_mic1.grid(True, alpha=0.3)
            
            # Barra de color para el primer espectrograma
            cbar1 = plt.colorbar(self.mic1_spec_image, ax=self.ax_mic1)
            cbar1.set_label('Intensidad (dB)')
            
            # Espectrograma de la señal beamformada - ABAJO
            self.ax_beam = fig.add_subplot(gs[1])
            
            # Inicializar con datos vacios (5 segundos)
            self.beam_spec_image = self.ax_beam.imshow(empty_spec, 
                                                      aspect='auto', cmap='viridis',
                                                      origin='lower', 
                                                      extent=[0, self.buffer_duration, 0, 4000])
            self.ax_beam.set_title('Señal Beamformada - Resultado', fontweight='bold')
            self.ax_beam.set_ylabel('Frecuencia (Hz)')
            self.ax_beam.set_xlabel('Tiempo (s)')
            self.ax_beam.set_ylim(0, 4000)
            self.ax_beam.set_xlim(0, self.buffer_duration)
            self.ax_beam.grid(True, alpha=0.3)
            
            # Barra de color para el espectrograma beamformed
            cbar2 = plt.colorbar(self.beam_spec_image, ax=self.ax_beam)
            cbar2.set_label('Intensidad (dB)')
            
            # Informacion de angulo - PARTE INFERIOR
            self.ax_info = fig.add_subplot(gs[2])
            self.ax_info.axis('off')
            self.angle_text = self.ax_info.text(0.5, 0.5, f'Angulo de Beamforming: {self.current_angle:.1f}°', 
                                              transform=self.ax_info.transAxes, ha='center', va='center',
                                              fontsize=12, fontweight='bold',
                                              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            
            # Boton para guardar audio
            self.btn_guardar_ax = plt.axes([0.8, 0.02, 0.15, 0.04])
            self.btn_guardar = plt.Button(self.btn_guardar_ax, 'Guardar Audio', color='lightgreen')
            self.btn_guardar.on_clicked(lambda x: self.guardar_audio_beamformed())
            
            plt.tight_layout()
            return True
            
        except Exception as e:
            print(f"Error configurando visualizacion beamforming: {e}")
            return False

    def update_plot(self, frame):
        """Actualiza los espectrogramas en tiempo real"""
        if not self.is_active or not self.is_processing:
            return [self.mic1_spec_image, self.beam_spec_image, self.angle_text]
        
        try:
            # Usar buffer completo de 5 segundos
            if len(self.audio_buffer) >= 512:  # Minimo para espectrograma
                
                # Espectrograma del primer microfono (canal 2)
                mic1_data = self.audio_buffer[:, 0]  # Primer microfono - todo el buffer
                f_mic1, t_mic1, Sxx_mic1 = self.compute_spectrogram(mic1_data)
                
                # Actualizar espectrograma del microfono 1
                self.mic1_spec_image.set_data(Sxx_mic1)
                self.mic1_spec_image.set_extent([0, self.buffer_duration, f_mic1[0], f_mic1[-1]])
                vmin_mic1, vmax_mic1 = np.percentile(Sxx_mic1, [5, 95])
                self.mic1_spec_image.set_clim(vmin_mic1, vmax_mic1)
                
                # Espectrograma de la señal beamformada
                beamformed_data = self.beamformed_buffer  # Todo el buffer
                f_beam, t_beam, Sxx_beam = self.compute_spectrogram(beamformed_data)
                
                # Actualizar espectrograma beamformed
                self.beam_spec_image.set_data(Sxx_beam)
                self.beam_spec_image.set_extent([0, self.buffer_duration, f_beam[0], f_beam[-1]])
                vmin_beam, vmax_beam = np.percentile(Sxx_beam, [5, 95])
                self.beam_spec_image.set_clim(vmin_beam, vmax_beam)
                
                # Actualizar texto del angulo
                self.angle_text.set_text(f'Angulo de Beamforming: {self.current_angle:.1f}°')
            
        except Exception as e:
            print(f"Error actualizando espectrogramas: {e}")
        
        return [self.mic1_spec_image, self.beam_spec_image, self.angle_text]

    def iniciar_beamforming(self):
        """Inicia el procesamiento de beamforming"""
        if self.is_active:
            print("Beamforming ya esta activo")
            return
            
        try:
            self.is_active = True
            self.is_processing = True
            
            print("BEAMFORMING ACTIVADO")
            print(f" - Blocksize: {self.blocksize} muestras")
            print(f" - Ventana visualizacion: {self.buffer_duration} segundos")
            print(" - Siguiendo automaticamente al DOA")
            print(" - Audio se guardara en formato WAV")
            print(f" - Ruta de guardado: {self.output_folder}")
            
        except Exception as e:
            print(f"Error iniciando beamforming: {e}")
            self.is_active = False

    def detener_beamforming(self):
        """Detiene el procesamiento de beamforming"""
        # Guardar automaticamente al detener
        if len(self.full_beamformed_audio) > 0:
            print("Guardando audio beamformed final...")
            self.guardar_audio_beamformed()
        
        self.is_processing = False
        self.is_active = False
            
        print("Beamforming detenido")