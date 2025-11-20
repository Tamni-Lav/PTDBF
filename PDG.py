"""
PDG.py - Visualizacion de señales de entrada
vcaa
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import time
import os
from scipy.io.wavfile import write

class MicrophoneArrayRealtime:
    def __init__(self, gestion_audio):
        self.gestion_audio = gestion_audio
        self.sample_rate = 16000
        self.channels = 4
        
        # Buffers
        self.audio_buffer = []
        self.full_audio_buffer = []
        self.running = True
        
        # Configuracion de graficos
        self.fig = None
        self.axs = None
        self.lines = []
        self.window_duration = 3.0
        self.window_samples = int(self.window_duration * self.sample_rate)
        
        self.amplification_factor = 15.0
        
        # Configuracion de guardado
        self.output_folder = r"C:\Users\leona\Desktop\TLTech\PFJdN\Audios_Crudos"
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Registrarse para recibir audio
        self.gestion_audio.agregar_suscriptor(self.recibir_audio)
        
        print(f"PDG inicializado - Amplificacion: {self.amplification_factor}x")

    def recibir_audio(self, audio_data):
        """Callback que recibe audio del gestor central"""
        if self.running and audio_data is not None:
            amplified_data = audio_data[:, 1:5].copy()
            
            for ch in range(4):
                amplified_data[:, ch] = amplified_data[:, ch] * self.amplification_factor
            
            self.audio_buffer.append(amplified_data)
            self.full_audio_buffer.append(audio_data[:, 1:5].copy())

    def setup_graficos(self):
        """Configuracion de graficos - Microfonos 1,2,3,4"""
        self.fig, self.axs = plt.subplots(4, 1, figsize=(14, 10))
        self.fig.suptitle(f'Señales de Entrada - Microfonos 1,2,3,4', 
                         fontsize=14, fontweight='bold')
        
        colores = ['blue', 'red', 'green', 'orange']
        
        # Configurar los 4 microfonos
        for i in range(4):
            mic_numero = i + 1
            self.axs[i].set_title(f"Microfono {mic_numero}", fontweight='bold', fontsize=11)
            self.axs[i].set_ylabel('Amplitud', fontsize=9)
            
            self.axs[i].set_ylim(-1.0, 1.0)
            self.axs[i].set_xlim(0, self.window_duration)
            self.axs[i].grid(True, alpha=0.3)
            self.axs[i].tick_params(labelsize=8)
            
            if i == 3:
                self.axs[i].set_xlabel('Tiempo (segundos)', fontsize=10)
            else:
                self.axs[i].set_xticklabels([])
            
            line, = self.axs[i].plot([], [], color=colores[i], linewidth=1.5)
            self.lines.append(line)

        # Botones
        button_guardar_ax = plt.axes([0.75, 0.01, 0.12, 0.04])
        self.button_guardar = Button(button_guardar_ax, "Guardar Audio", color='lightgreen')
        self.button_guardar.on_clicked(self.guardar_audio_completo)

        button_parar_ax = plt.axes([0.88, 0.01, 0.10, 0.04])
        self.button_parar = Button(button_parar_ax, "Cerrar", color='lightcoral')
        self.button_parar.on_clicked(self.detener_visualizacion)

        plt.tight_layout()

    def update_plot(self, frame):
        """Actualiza graficos en tiempo real"""
        if not self.running or len(self.audio_buffer) == 0:
            return self.lines

        try:
            if len(self.audio_buffer) > 0:
                all_data = np.concatenate(self.audio_buffer, axis=0)
                
                if len(all_data) > self.window_samples:
                    all_data = all_data[-self.window_samples:]
                
                current_time = np.linspace(max(0, self.window_duration - len(all_data)/self.sample_rate), 
                                         self.window_duration, len(all_data))
                
                # Actualizar microfonos 1,2,3,4
                for i in range(4):
                    if all_data.shape[1] > i:
                        self.lines[i].set_data(current_time, all_data[:, i])
                    
        except Exception as e:
            print(f"Error actualizando graficos PDG: {e}")

        return self.lines

    def guardar_audio_completo(self, event=None):
        """Guarda el audio completo de los microfonos 1,2,3,4"""
        print("\nGuardando audio desde PDG...")
        
        if len(self.full_audio_buffer) == 0:
            print("Error: No hay datos de audio para guardar")
            return

        try:
            audio_data = np.concatenate(self.full_audio_buffer, axis=0)
            audio_int16 = np.int16(audio_data * 32767)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Guardar microfonos individuales
            for i in range(4):
                mic_numero = i + 1
                filepath = os.path.join(self.output_folder, f"mic_{mic_numero}_{timestamp}.wav")
                write(filepath, self.sample_rate, audio_int16[:, i])
                print(f"Guardado: {filepath}")

            # Guardar todos los canales
            all_channels_path = os.path.join(self.output_folder, f"todos_canales_{timestamp}.wav")
            write(all_channels_path, self.sample_rate, audio_int16)
            print(f"Guardado multicanales: {all_channels_path}")

            duracion_total = len(audio_data) / self.sample_rate
            print(f"Duracion TOTAL: {duracion_total:.2f} segundos")
            print(f"Amplificacion aplicada: {self.amplification_factor}x")
            
        except Exception as e:
            print(f"Error guardando audio PDG: {e}")

    def detener_visualizacion(self, event=None):
        """Detiene la visualizacion"""
        print("\nCerrando PDG...")
        self.running = False
        if self.fig:
            plt.close(self.fig)