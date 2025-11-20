"""
Beamforming.py - Sistema de beamforming con visualización completa.
CON FILTRO IIR Y 4 GRÁFICAS - CANAL 0 vs BEAMFORMED (AMPLIFICADO 30x)
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
    """Sistema de beamforming con visualización completa de señales."""
    
    def __init__(self, gestion_audio, doa_system):
        """Inicializa el sistema de beamforming.
        
        Args:
            gestion_audio: Gestor central de audio
            doa_system: Sistema DOA para obtener dirección
        """
        self.gestion_audio = gestion_audio
        self.doa = doa_system
        
        # Configuración de audio
        self.sample_rate = 16000
        self.blocksize = 1024
        
        # Configuración del array
        self.radio = 0.0325
        self.mic_positions = np.array([
            [-self.radio, 0],    # Micrófono 1
            [0, -self.radio],    # Micrófono 2  
            [self.radio, 0],     # Micrófono 3
            [0, self.radio]      # Micrófono 4
        ])
        
        self.sound_speed = 343.0
        
        # Estado del sistema
        self.is_active = False
        self.is_processing = False
        self.current_angle = 0
        
        # Buffers para visualización (5 segundos)
        self.buffer_duration = 5
        self.buffer_size = self.buffer_duration * self.sample_rate
        
        # Buffers para CANAL 0 y Beamformed
        self.canal0_buffer = np.zeros(self.buffer_size)
        self.beamformed_buffer = np.zeros(self.buffer_size)
        self.full_beamformed_audio = []
        
        # Configuración de filtro IIR
        self.filtro_iir_b, self.filtro_iir_a = self._diseñar_filtro_iir()
        print(f"Filtro IIR diseñado - Orden: {len(self.filtro_iir_b)-1}")
        
        # Estado del filtro
        self.filtro_zi = None
        
        # Configuración de guardado
        self.output_folder = r"C:\Users\leona\Desktop\TLTech\PFJdN\Audios_Beamformed"
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Pre-cálculos
        self.delays_precalculated = self._precalculate_all_delays()
        
        # Visualización
        self.fig = None
        self.ax_temporal = None
        self.ax_espectral = None
        self.ax_canal0_spec = None
        self.ax_beam_spec = None
        self.ax_info = None
        
        self.canal0_time_line = None
        self.beam_time_line = None
        self.canal0_freq_line = None
        self.beam_freq_line = None
        self.canal0_spec_image = None
        self.beam_spec_image = None
        self.angle_text = None
        
        # Registrarse para recibir audio
        self.gestion_audio.agregar_suscriptor(self.recibir_audio)
        
        print("Beamforming inicializado - CANAL 0 vs BEAMFORMED (AMPLIFICADO 30x)")
        print(f" - Comparación: Canal 0 vs Señal Beamformed")
        print(f" - Amplificación Beamformed: 30x en ambos dominios")
        print(f" - Duración visualización: {self.buffer_duration} segundos")

    def _diseñar_filtro_iir(self):
        """Diseña filtro IIR Butterworth pasa-banda para voz."""
        orden = 4
        lowcut = 300
        highcut = 3400
        
        b, a = signal.butter(
            N=orden,
            Wn=[lowcut, highcut],
            btype='band',
            fs=self.sample_rate,
            analog=False
        )
        
        return b, a

    def _aplicar_filtro_iir(self, señal, usar_filtfilt=True):
        """Aplica filtro IIR a la señal."""
        if usar_filtfilt:
            return signal.filtfilt(self.filtro_iir_b, self.filtro_iir_a, señal)
        else:
            if self.filtro_zi is None:
                self.filtro_zi = signal.lfilter_zi(self.filtro_iir_b, self.filtro_iir_a)
            
            señal_filtrada, self.filtro_zi = signal.lfilter(
                self.filtro_iir_b, self.filtro_iir_a, señal, zi=self.filtro_zi
            )
            return señal_filtrada

    def _precalculate_all_delays(self):
        """Pre-calcula todos los delays para 360 grados."""
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
        """Obtiene delays pre-calculados para un ángulo específico."""
        angle_idx = int(round(angle_deg)) % 360
        return self.delays_precalculated[angle_idx].copy()

    def apply_beamforming(self, audio_data, angle_deg):
        """Aplica beamforming con ventaneo y filtrado IIR."""
        # VENTANEO para mejorar calidad del beamforming
        ventana = np.hanning(len(audio_data))
        audio_ventaneado = audio_data * ventana[:, np.newaxis]
        
        delays = self.calculate_delays(angle_deg)
        delay_samples = np.round(delays * self.sample_rate).astype(int)
        max_delay = np.max(delay_samples)
        
        output_length = len(audio_ventaneado) + max_delay
        aligned_signals = np.zeros((output_length, 4))
        
        for mic in range(4):
            start_idx = delay_samples[mic]
            end_idx = start_idx + len(audio_ventaneado)
            if end_idx <= output_length:
                aligned_signals[start_idx:end_idx, mic] = audio_ventaneado[:, mic]
        
        beamformed_signal = np.sum(aligned_signals, axis=1) / 4.0
        señal_sin_filtrar = beamformed_signal[:len(audio_ventaneado)]
        
        # FILTRADO IIR para mejorar calidad vocal
        señal_filtrada = self._aplicar_filtro_iir(señal_sin_filtrar, usar_filtfilt=True)
        
        return señal_filtrada

    def recibir_audio(self, audio_data):
        """Callback que recibe audio del gestor central."""
        if self.is_active and audio_data is not None and self.is_processing:
            try:
                if audio_data.shape[1] >= 6:
                    # Usar CANAL 0 para comparación
                    canal0_signal = audio_data[:, 0].copy()
                    
                    # Actualizar buffer del CANAL 0
                    self.canal0_buffer = np.roll(self.canal0_buffer, -len(canal0_signal))
                    self.canal0_buffer[-len(canal0_signal):] = canal0_signal
                    
                    # Obtener ángulo actual del DOA
                    self.current_angle = self.doa.angulo_actual
                    
                    # Usar canales 1-4 para beamforming
                    mic_data = audio_data[:, 1:5].copy()
                    
                    # Aplicar beamforming CON FILTRO IIR
                    beamformed = self.apply_beamforming(mic_data, self.current_angle)
                    
                    # *** AMPLIFICACIÓN 30x PARA MEJOR VISUALIZACIÓN ***
                    factor_amplificacion = 30.0
                    beamformed_amplificado = beamformed * factor_amplificacion
                    
                    # Actualizar buffer beamformed CON AMPLIFICACIÓN 30x
                    self.beamformed_buffer = np.roll(self.beamformed_buffer, -len(beamformed_amplificado))
                    self.beamformed_buffer[-len(beamformed_amplificado):] = beamformed_amplificado
                    
                    # DEBUG: Verificar niveles ocasionalmente
                    if np.random.random() < 0.05:
                        rms_canal0 = np.sqrt(np.mean(canal0_signal**2))
                        rms_beam = np.sqrt(np.mean(beamformed**2))
                        rms_beam_amp = np.sqrt(np.mean(beamformed_amplificado**2))
                        print(f"Amplificación 30x - Canal0: {rms_canal0:.4f}, Beam: {rms_beam:.4f}, Beam(x30): {rms_beam_amp:.4f}")
                    
                    # Guardar en buffer completo SIN amplificación
                    self.full_beamformed_audio.append(beamformed.copy())
                        
            except Exception as e:
                print(f"Error en beamforming: {e}")

    def compute_spectrogram(self, data, nperseg=512):
        """Calcula el espectrograma de una señal."""
        f, t, Sxx = signal.spectrogram(data, self.sample_rate, nperseg=nperseg, noverlap=nperseg//2)
        return f, t, 10 * np.log10(Sxx + 1e-10)

    def compute_spectrum(self, data):
        """Calcula el espectro de frecuencia de una señal."""
        fft_data = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), 1/self.sample_rate)
        magnitude = np.abs(fft_data)
        # Tomar solo frecuencias positivas
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]
        return positive_freqs, positive_magnitude

    def guardar_audio_beamformed(self):
        """Guarda el audio beamformed completo en archivo WAV."""
        if len(self.full_beamformed_audio) == 0:
            print("No hay datos de audio beamformed para guardar")
            return

        try:
            audio_data = np.concatenate(self.full_beamformed_audio, axis=0)
            audio_int16 = np.int16(audio_data * 32767)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"beamformed_iir_{timestamp}.wav"
            filepath = os.path.join(self.output_folder, filename)
            
            wavfile.write(filepath, self.sample_rate, audio_int16)
            
            duracion = len(audio_data) / self.sample_rate
            print(f"Audio beamformed CON FILTRO IIR guardado: {filepath}")
            print(f"Duración: {duracion:.2f} segundos")
            
            self.full_beamformed_audio = []
            
        except Exception as e:
            print(f"Error guardando audio beamformed: {e}")

    def configurar_visualizacion(self, fig):
        """Configura las 4 gráficas en una sola ventana."""
        try:
            self.fig = fig
            self.fig.clear()
            
            # Crear layout 2x2
            gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.15])
            
            # Título principal
            self.fig.suptitle(
                f'BEAMFORMING - CANAL 0 vs BEAMFORMED (AMPLIFICADO 30x)', 
                fontsize=14, 
                fontweight='bold'
            )
            
            # 1. GRÁFICA TEMPORAL (Arriba-Izquierda)
            self.ax_temporal = fig.add_subplot(gs[0, 0])
            self.ax_temporal.set_title('DOMINIO TEMPORAL - Canal 0 vs Beamformed (Amplificado 30x)', 
                                     fontweight='bold', fontsize=10)
            self.ax_temporal.set_ylabel('Amplitud')
            self.ax_temporal.set_xlabel('Tiempo (s)')
            self.ax_temporal.grid(True, alpha=0.3)
            self.ax_temporal.set_xlim(0, self.buffer_duration)
            self.ax_temporal.set_ylim(-3.0, 3.0)  # Aumentado para 30x
            
            # Crear líneas para temporal
            tiempo = np.linspace(0, self.buffer_duration, self.buffer_size)
            self.canal0_time_line, = self.ax_temporal.plot(tiempo, self.canal0_buffer, 'b-', 
                                                         linewidth=1.2, alpha=0.9, label='Canal 0 (Original)')
            self.beam_time_line, = self.ax_temporal.plot(tiempo, self.beamformed_buffer, 'r-', 
                                                       linewidth=1.2, alpha=0.9, label='Beamformed (Amplificado 30x)')
            self.ax_temporal.legend(loc='upper right', fontsize=8)
            
            # 2. GRÁFICA ESPECTRAL (Arriba-Derecha)
            self.ax_espectral = fig.add_subplot(gs[0, 1])
            self.ax_espectral.set_title('DOMINIO FRECUENCIA - Canal 0 vs Beamformed (Amplificado 30x)', 
                                      fontweight='bold', fontsize=10)
            self.ax_espectral.set_ylabel('Amplitud (Normalizada)')
            self.ax_espectral.set_xlabel('Frecuencia (Hz)')
            self.ax_espectral.grid(True, alpha=0.3)
            self.ax_espectral.set_xlim(0, 4000)
            self.ax_espectral.set_ylim(0, 1.2)
            
            # Crear líneas para espectral
            freqs = np.linspace(0, 4000, 1000)
            self.canal0_freq_line, = self.ax_espectral.plot(freqs, np.zeros_like(freqs), 'b-', 
                                                          linewidth=2.0, alpha=0.9, label='Canal 0 (Original)')
            self.beam_freq_line, = self.ax_espectral.plot(freqs, np.zeros_like(freqs), 'r-', 
                                                        linewidth=2.0, alpha=0.9, label='Beamformed (Amplificado 30x)')
            self.ax_espectral.legend(loc='upper right', fontsize=8)
            
            # 3. ESPECTROGRAMA CANAL 0 (Abajo-Izquierda)
            self.ax_canal0_spec = fig.add_subplot(gs[1, 0])
            empty_spec = np.zeros((100, 100))
            self.canal0_spec_image = self.ax_canal0_spec.imshow(
                empty_spec, 
                aspect='auto', 
                cmap='viridis',
                origin='lower', 
                extent=[0, self.buffer_duration, 0, 4000]
            )
            self.ax_canal0_spec.set_title('ESPECTROGRAMA - Canal 0 Original', fontweight='bold', fontsize=10)
            self.ax_canal0_spec.set_ylabel('Frecuencia (Hz)')
            self.ax_canal0_spec.set_xlabel('Tiempo (s)')
            self.ax_canal0_spec.grid(True, alpha=0.3)
            plt.colorbar(self.canal0_spec_image, ax=self.ax_canal0_spec, label='dB')
            
            # 4. ESPECTROGRAMA BEAMFORMED (Abajo-Derecha)
            self.ax_beam_spec = fig.add_subplot(gs[1, 1])
            self.beam_spec_image = self.ax_beam_spec.imshow(
                empty_spec, 
                aspect='auto', 
                cmap='viridis',
                origin='lower', 
                extent=[0, self.buffer_duration, 0, 4000]
            )
            self.ax_beam_spec.set_title('ESPECTROGRAMA - Señal Beamformed + Filtro IIR (Amplificado 30x)', 
                                      fontweight='bold', fontsize=10)
            self.ax_beam_spec.set_ylabel('Frecuencia (Hz)')
            self.ax_beam_spec.set_xlabel('Tiempo (s)')
            self.ax_beam_spec.grid(True, alpha=0.3)
            plt.colorbar(self.beam_spec_image, ax=self.ax_beam_spec, label='dB')
            
            # 5. INFORMACIÓN (Fila inferior completa)
            self.ax_info = fig.add_subplot(gs[2, :])
            self.ax_info.axis('off')
            self.angle_text = self.ax_info.text(
                0.5, 0.5, 
                f'COMPARACIÓN: Canal 0 vs Beamformed (AMPLIFICADO 30x) | Ángulo: {self.current_angle:.1f}° | Filtro IIR: 300-3400 Hz', 
                transform=self.ax_info.transAxes, 
                ha='center', va='center',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue")
            )
            
            # Botón para guardar audio
            self.btn_guardar_ax = plt.axes([0.8, 0.02, 0.15, 0.04])
            self.btn_guardar = plt.Button(self.btn_guardar_ax, 'Guardar Audio', color='lightgreen')
            self.btn_guardar.on_clicked(lambda x: self.guardar_audio_beamformed())
            
            plt.tight_layout()
            return True
            
        except Exception as e:
            print(f"Error configurando visualización: {e}")
            return False

    def update_plot(self, frame):
        """Actualiza las 4 gráficas en tiempo real."""
        if not self.is_active or not self.is_processing:
            return [self.canal0_time_line, self.beam_time_line, self.canal0_freq_line, self.beam_freq_line, 
                   self.canal0_spec_image, self.beam_spec_image, self.angle_text]
        
        try:
            # Verificar que los buffers tienen datos
            if len(self.canal0_buffer) == 0 or len(self.beamformed_buffer) == 0:
                return [self.canal0_time_line, self.beam_time_line, self.canal0_freq_line, self.beam_freq_line, 
                       self.canal0_spec_image, self.beam_spec_image, self.angle_text]
            
            # Crear array de tiempo
            tiempo = np.linspace(0, self.buffer_duration, len(self.canal0_buffer))
            
            # Actualizar líneas temporales
            self.canal0_time_line.set_data(tiempo, self.canal0_buffer)
            self.beam_time_line.set_data(tiempo, self.beamformed_buffer)
            
            # Actualizar gráfica espectral
            if np.max(np.abs(self.canal0_buffer)) > 0.001:
                freqs_canal0, spectrum_canal0 = self.compute_spectrum(self.canal0_buffer)
                freqs_beam, spectrum_beam = self.compute_spectrum(self.beamformed_buffer)
                
                # Normalizar para visualización
                max_spectrum = max(np.max(spectrum_canal0), np.max(spectrum_beam))
                if max_spectrum > 0:
                    spectrum_canal0_norm = spectrum_canal0 / max_spectrum
                    spectrum_beam_norm = spectrum_beam / max_spectrum
                else:
                    spectrum_canal0_norm = spectrum_canal0
                    spectrum_beam_norm = spectrum_beam
                
                self.canal0_freq_line.set_data(freqs_canal0, spectrum_canal0_norm)
                self.beam_freq_line.set_data(freqs_beam, spectrum_beam_norm)
            
            # Actualizar espectrogramas
            if len(self.canal0_buffer) >= 512:
                # Espectrograma CANAL 0
                f_canal0, t_canal0, Sxx_canal0 = self.compute_spectrogram(self.canal0_buffer)
                self.canal0_spec_image.set_data(Sxx_canal0)
                self.canal0_spec_image.set_extent([0, self.buffer_duration, f_canal0[0], f_canal0[-1]])
                vmin_canal0, vmax_canal0 = np.percentile(Sxx_canal0, [5, 95])
                self.canal0_spec_image.set_clim(vmin_canal0, vmax_canal0)
                
                # Espectrograma beamformed
                f_beam, t_beam, Sxx_beam = self.compute_spectrogram(self.beamformed_buffer)
                self.beam_spec_image.set_data(Sxx_beam)
                self.beam_spec_image.set_extent([0, self.buffer_duration, f_beam[0], f_beam[-1]])
                vmin_beam, vmax_beam = np.percentile(Sxx_beam, [5, 95])
                self.beam_spec_image.set_clim(vmin_beam, vmax_beam)
            
            # Actualizar texto del ángulo
            self.angle_text.set_text(f'COMPARACIÓN: Canal 0 vs Beamformed (AMPLIFICADO 30x) | Ángulo: {self.current_angle:.1f}° | Filtro IIR: 300-3400 Hz')
            
        except Exception as e:
            print(f"Error actualizando gráficas: {e}")
        
        return [self.canal0_time_line, self.beam_time_line, self.canal0_freq_line, self.beam_freq_line, 
               self.canal0_spec_image, self.beam_spec_image, self.angle_text]

    def iniciar_beamforming(self):
        """Inicia el procesamiento de beamforming."""
        if self.is_active:
            print("Beamforming ya está activo")
            return
            
        try:
            self.is_active = True
            self.is_processing = True
            
            print("BEAMFORMING ACTIVADO - SEÑAL BEAMFORMED AMPLIFICADA 30x")
            print(f" - Canal 0: Señal original de referencia")
            print(f" - Beamformed: Procesada con filtro IIR 300-3400 Hz")
            print(f" - Amplificación visual: 30x en ambos dominios")
            print(f" - Audio guardado: Sin amplificación (calidad original)")
            print(f" - Ventana visualización: {self.buffer_duration} segundos")
            
        except Exception as e:
            print(f"Error iniciando beamforming: {e}")
            self.is_active = False

    def detener_beamforming(self):
        """Detiene el procesamiento de beamforming."""
        if len(self.full_beamformed_audio) > 0:
            print("Guardando audio beamformed final...")
            self.guardar_audio_beamformed()
        
        self.is_processing = False
        self.is_active = False
        self.filtro_zi = None
            
        print("Beamforming detenido")