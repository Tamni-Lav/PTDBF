"""
Beamforming.py - Versión OPTIMIZADA para DOA ULTRA ESTABLE
CORREGIDO: Sin petardeos - Optimizado para ángulos estables
"""

import numpy as np
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
        
        # Configuración
        self.sample_rate = 16000
        self.blocksize = 1024
        
        # Array configuration
        self.radio = 0.0325
        self.mic_positions = np.array([
            [0, -self.radio],    # Mic 1 - Canal 2
            [self.radio, 0],    # Mic 2 - Canal 3  
            [0, self.radio],     # Mic 3 - Canal 4
            [-self.radio, 0]      # Mic 4 - Canal 5
        ])
        self.sound_speed = 343.0
        
        # Estado
        self.is_active = False
        self.is_processing = False
        self.current_angle = 0
        
        # ✅ OPTIMIZADO PARA DOA ESTABLE - MENOS SUAVIZADO NECESARIO
        self.angle_confidence = 0.0
        self.consecutive_stable_frames = 0
        
        # ✅ GANANCIAS OPTIMIZADAS - MÁS CONSERVADORAS
        self.ganancia_base = 1.8  # Ajustado para DOA estable
        self.ganancia_visual = 2.2
        self.umbral_compresor = 0.18
        self.ratio_compresion = 2.2
        
        # Buffers
        self.buffer_duration = 5
        self.buffer_size = self.buffer_duration * self.sample_rate
        self.canal0_buffer = np.zeros(self.buffer_size)
        self.beamformed_buffer = np.zeros(self.buffer_size)
        self.beamformed_filtrado_buffer = np.zeros(self.buffer_size)
        
        # Audio para guardar
        self.full_beamformed_audio = []
        self.buffer_count = 0
        
        # Configuración
        self.output_folder = r"C:\Users\leona\Desktop\TLTech\PFJdN\Audios_Beamformed"
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Pre-cálculos OPTIMIZADOS
        self.delays_precalculated = self._precalculate_all_delays_con_fracciones()
        
        # ✅ FILTRO PASABANDA 50Hz - 7000Hz (Orden 5)
        self.filtro_pasabanda_b, self.filtro_pasabanda_a = self._crear_filtro_pasabanda()
        
        # Visualización
        self.fig = None
        self.axes = None
        self.lineas_temporales = [None, None, None]
        self.imagenes_espectrograma = [None, None, None]
        self.lineas_espectro = [None, None, None]
        
        # BOTÓN
        self.btn_guardar = None
        
        # Estado del compresor
        self.compression_state = 1.0
        
        # Registro
        self.gestion_audio.agregar_suscriptor(self.recibir_audio)
        
        print("✅ BEAMFORMING OPTIMIZADO PARA DOA ULTRA ESTABLE")
        print("   - Filtro: 50Hz - 7000Hz (orden 5)")
        print("   - Adaptado a ángulos enteros estables")
        print("   - Ganancia adaptativa por confianza DOA")

    def _crear_filtro_pasabanda(self):
        """Crea filtro pasabanda IIR de 50Hz a 7000Hz, orden 5"""
        try:
            nyquist = self.sample_rate / 2.0
            low_freq = 50.0 / nyquist
            high_freq = 7000.0 / nyquist
            
            if low_freq <= 0 or high_freq >= 1 or low_freq >= high_freq:
                low_freq = 50.0 / nyquist
                high_freq = 7000.0 / nyquist
            
            b, a = signal.butter(5, [low_freq, high_freq], btype='band', analog=False)
            print(f"✅ Filtro pasabanda creado: 50Hz - 7000Hz, orden 5")
            return b, a
            
        except Exception as e:
            print(f"❌ Error creando filtro pasabanda: {e}")
            return [1.0], [1.0]

    def aplicar_filtro_pasabanda(self, señal):
        """Aplica el filtro pasabanda 50Hz-7000Hz"""
        try:
            if len(self.filtro_pasabanda_b) == 1 and self.filtro_pasabanda_b[0] == 1.0:
                return señal
            return signal.filtfilt(self.filtro_pasabanda_b, self.filtro_pasabanda_a, señal)
        except Exception as e:
            print(f"⚠️ Error aplicando filtro pasabanda: {e}")
            return señal

    def _precalculate_all_delays_con_fracciones(self):
        """Precalcula delays con parte fraccionaria"""
        delays = np.zeros((360, 4, 2))
        
        for angle_deg in range(360):
            angle_rad = np.deg2rad(angle_deg)
            direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            
            for mic in range(4):
                distance = np.dot(self.mic_positions[mic], direction)
                delay_seconds = distance / self.sound_speed
                delay_samples = delay_seconds * self.sample_rate
                
                delays[angle_deg, mic, 0] = int(np.floor(delay_samples))
                delays[angle_deg, mic, 1] = delay_samples - delays[angle_deg, mic, 0]
            
            min_delay = np.min(delays[angle_deg, :, 0])
            delays[angle_deg, :, 0] -= min_delay
            
        return delays

    def apply_beamforming_optimized(self, audio_data, angle_deg, confidence):
        """Beamforming OPTIMIZADO para DOA estable"""
        # ✅ USAR SOLO DELAYS ENTEROS (más estable)
        angle_deg_int = int(angle_deg) % 360
        delays = self.delays_precalculated[angle_deg_int]
        
        max_delay = int(np.max(delays[:, 0]))
        output_length = len(audio_data) + max_delay
        
        beamformed = np.zeros(output_length)
        weights = np.zeros(output_length)
        
        for mic in range(4):
            delay_int = int(delays[mic, 0])
            
            start_idx = delay_int
            end_idx = start_idx + len(audio_data)
            
            if end_idx <= output_length:
                beamformed[start_idx:end_idx] += audio_data[:, mic]
                weights[start_idx:end_idx] += 1.0
        
        weights[weights == 0] = 1.0
        beamformed = beamformed / weights
        beamformed = beamformed[:len(audio_data)]
        
        # ✅ GANANCIA ADAPTATIVA SEGÚN CONFIANZA DOA
        adaptive_gain = 1.0
        
        if confidence > 0.7:
            # Alta confianza - ganancia normal
            adaptive_gain = 1.0
            self.consecutive_stable_frames += 1
        elif confidence > 0.4:
            # Confianza media - ganancia reducida
            adaptive_gain = 0.8
            self.consecutive_stable_frames = max(0, self.consecutive_stable_frames - 1)
        else:
            # Baja confianza - ganancia muy reducida
            adaptive_gain = 0.6
            self.consecutive_stable_frames = 0
        
        # ✅ BONUS POR ESTABILIDAD PROLONGADA
        if self.consecutive_stable_frames > 20:
            adaptive_gain *= 1.1  # Pequeño bonus por estabilidad
        
        beamformed = beamformed * adaptive_gain
        
        # ✅ NORMALIZACIÓN CONSERVADORA
        max_val = np.max(np.abs(beamformed))
        if max_val > 0.25:
            beamformed = beamformed * (0.25 / max_val)
        
        return beamformed

    def aplicar_compresor_optimizado(self, señal, umbral=0.18, ratio=2.2):
        """Compresor OPTIMIZADO para señales estables"""
        señal_comprimida = np.zeros_like(señal)
        
        # ✅ VENTANA MÁS LARGA PARA MÁS SUAVIDAD
        window_size = 100
        envelope = np.convolve(np.abs(señal), np.ones(window_size)/window_size, mode='same')
        
        for i in range(len(señal)):
            nivel = envelope[i]
            
            if nivel > umbral:
                exceso_db = 20 * np.log10(nivel / umbral)
                reduccion_db = exceso_db * (1 - 1/ratio)
                ganancia_objetivo = 10 ** (-reduccion_db / 20)
            else:
                ganancia_objetivo = 1.0
            
            # ✅ SUAVIDAD EXTREMA EN CAMBIOS
            self.compression_state = 0.99 * self.compression_state + 0.01 * ganancia_objetivo
            señal_comprimida[i] = señal[i] * self.compression_state
        
        return señal_comprimida

    def suavizar_transicion_minima(self, señal, muestras_suavizado=32):
        """Suavizado MÍNIMO - DOA ya es estable"""
        if len(señal) < muestras_suavizado * 2:
            return señal
            
        señal_suavizada = señal.copy()
        
        # ✅ SOLO SUAVIDAD EN BORDES (menos procesamiento)
        ventana_final = 0.5 - 0.5 * np.cos(np.linspace(np.pi, 0, muestras_suavizado))
        señal_suavizada[-muestras_suavizado:] *= ventana_final
        
        return señal_suavizada

    def recibir_audio(self, audio_data):
        """Callback OPTIMIZADO para DOA estable"""
        if not self.is_active or audio_data is None or not self.is_processing:
            return
            
        try:
            if audio_data.shape[1] >= 6:
                canal0_signal = audio_data[:, 0].copy()
                
                self.canal0_buffer = np.roll(self.canal0_buffer, -len(canal0_signal))
                self.canal0_buffer[-len(canal0_signal):] = canal0_signal
                
                # ✅ DOA YA VIENE ESTABLE - SOLO USAR ÁNGULO
                self.current_angle, self.angle_confidence = self.doa.get_angulo_actual()
                
                mic_data = audio_data[:, 1:5].copy()
                
                # ✅ BEAMFORMING CON CONFIANZA INTEGRADA
                beamformed = self.apply_beamforming_optimized(
                    mic_data, self.current_angle, self.angle_confidence
                )
                
                # ✅ CADENA DE PROCESAMIENTO SIMPLIFICADA
                beamformed_ganancia = beamformed * self.ganancia_base
                beamformed_comprimido = self.aplicar_compresor_optimizado(beamformed_ganancia)
                beamformed_filtrado = self.aplicar_filtro_pasabanda(beamformed_comprimido)
                beamformed_final = self.suavizar_transicion_minima(beamformed_filtrado, 32)
                
                # Limpieza de datos
                beamformed_final = np.nan_to_num(beamformed_final, nan=0.0, posinf=0.0, neginf=0.0)
                
                # ✅ GUARDAR AUDIO
                self.full_beamformed_audio.append(beamformed_final.copy())
                self.buffer_count += 1
                
                # Para visualización
                beamformed_visual = beamformed * self.ganancia_visual
                beamformed_visual = np.clip(beamformed_visual, -1.0, 1.0)
                
                beamformed_filtrado_visual = beamformed_filtrado * self.ganancia_visual
                beamformed_filtrado_visual = np.clip(beamformed_filtrado_visual, -1.0, 1.0)
                
                # Actualizar buffers de visualización
                self.beamformed_buffer = np.roll(self.beamformed_buffer, -len(beamformed_visual))
                self.beamformed_buffer[-len(beamformed_visual):] = beamformed_visual
                
                self.beamformed_filtrado_buffer = np.roll(
                    self.beamformed_filtrado_buffer, -len(beamformed_filtrado_visual)
                )
                self.beamformed_filtrado_buffer[-len(beamformed_filtrado_visual):] = beamformed_filtrado_visual
                
                # ✅ MONITOREO MEJORADO
                if self.buffer_count % 40 == 0:
                    rms_beam = np.sqrt(np.mean(beamformed_final**2))
                    stability = "⚡" if self.consecutive_stable_frames > 20 else ""
                    print(f"🎯 Beam - Ángulo: {self.current_angle}° "
                          f"Conf: {self.angle_confidence:.2f} "
                          f"RMS: {rms_beam:.3f} {stability}")
                        
        except Exception as e:
            print(f"❌ Error en beamforming: {e}")

    def guardar_audio_beamformed(self, event=None):
        """Guardado con información de estabilidad"""
        if len(self.full_beamformed_audio) == 0:
            print("❌ No hay datos de audio para guardar")
            return

        try:
            print("💾 Guardando audio beamformed...")
            audio_data = np.concatenate(self.full_beamformed_audio, axis=0)
            
            max_val = np.max(np.abs(audio_data))
            print(f"📊 Máximo en grabación: {max_val:.4f}")
            
            if max_val > 0.8:
                factor_ajuste = 0.8 / max_val
                audio_data = audio_data * factor_ajuste
                print(f"   🔧 Ajuste aplicado: {factor_ajuste:.3f}x")
            
            # Fade out suave
            fade_samples = min(256, len(audio_data) // 20)
            if len(audio_data) > fade_samples:
                fade_out = 0.5 - 0.5 * np.cos(np.linspace(np.pi, 0, fade_samples))
                audio_data[-fade_samples:] *= fade_out
            
            audio_int16 = np.int16(audio_data * 32767)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"beamformed_estable_{timestamp}.wav"
            filepath = os.path.join(self.output_folder, filename)
            
            wavfile.write(filepath, self.sample_rate, audio_int16)
            
            duracion = len(audio_data) / self.sample_rate
            print(f"✅ AUDIO GUARDADO: {filepath}")
            print(f"   Duración: {duracion:.2f}s")
            print(f"   Ángulo final: {self.current_angle}°")
            print(f"   Frames estables: {self.consecutive_stable_frames}")
            
            self.full_beamformed_audio = []
            self.buffer_count = 0
            
        except Exception as e:
            print(f"❌ Error guardando audio: {e}")

    def configurar_visualizacion(self, fig):
        """Configura la visualización 3x3"""
        try:
            self.fig = fig
            self.fig.clear()
            
            gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)
            
            self.fig.suptitle('BEAMFORMING - DOA ULTRA ESTABLE', 
                            fontsize=12, fontweight='bold')
            
            self.axes = np.zeros((3, 3), dtype=object)
            titulos_columnas = ['ORIGINAL', 'BEAMFORMED', 'BEAMFORMED + FILTRO']
            titulos_filas = ['Espectrograma', 'Señal Temporal', 'Espectro Frecuencia']
            
            for col in range(3):
                for row in range(3):
                    self.axes[row, col] = fig.add_subplot(gs[row, col])
                    
                    if row == 0:
                        self.axes[row, col].set_title(titulos_columnas[col], fontweight='bold')
                    
                    if col == 0:
                        self.axes[row, col].set_ylabel(titulos_filas[row], fontweight='bold')
                    
                    if row == 0:
                        self.axes[row, col].set_xlabel('Tiempo (s)')
                        self.axes[row, col].set_ylabel('Frecuencia (Hz)')
                    elif row == 1:
                        self.axes[row, col].set_xlabel('Tiempo (s)')
                        self.axes[row, col].set_ylabel('Amplitud')
                        self.axes[row, col].set_ylim(-1.0, 1.0)
                        self.axes[row, col].grid(True, alpha=0.3)
                    else:
                        self.axes[row, col].set_xlabel('Frecuencia (Hz)')
                        self.axes[row, col].set_ylabel('Magnitud (dB)')
                        self.axes[row, col].set_xlim(0, 8000)
                        self.axes[row, col].grid(True, alpha=0.3)
            
            tiempo = np.linspace(0, self.buffer_duration, self.buffer_size)
            colores = ['blue', 'green', 'red']
            
            for col in range(3):
                self.lineas_temporales[col], = self.axes[1, col].plot(
                    tiempo, np.zeros(self.buffer_size), 
                    color=colores[col], linewidth=1.0
                )
            
            for col in range(3):
                empty_spec = np.zeros((100, 100))
                self.imagenes_espectrograma[col] = self.axes[0, col].imshow(
                    empty_spec, aspect='auto', cmap='viridis',
                    origin='lower', extent=[0, self.buffer_duration, 0, 8000]
                )
                if col == 2:
                    plt.colorbar(self.imagenes_espectrograma[col], ax=self.axes[0, col])
            
            frecuencias = np.fft.rfftfreq(self.buffer_size, 1/self.sample_rate)
            for col in range(3):
                self.lineas_espectro[col], = self.axes[2, col].plot(
                    frecuencias, np.zeros_like(frecuencias) - 60,
                    color=colores[col], linewidth=1.5
                )
                if col == 2:
                    self.axes[2, col].axvline(50, color='red', linestyle='--', alpha=0.7, label='50Hz')
                    self.axes[2, col].axvline(7000, color='red', linestyle='--', alpha=0.7, label='7000Hz')
                    self.axes[2, col].legend(fontsize=8)
            
            from matplotlib.widgets import Button
            btn_ax = plt.axes([0.8, 0.01, 0.15, 0.04])
            self.btn_guardar = Button(btn_ax, 'Guardar Audio', color='lightgreen')
            self.btn_guardar.on_clicked(self.guardar_audio_beamformed)
            
            plt.tight_layout()
            return True
            
        except Exception as e:
            print(f"❌ Error en visualización: {e}")
            return False

    def update_plot(self, frame):
        """Actualiza la gráfica 3x3 en tiempo real"""
        if not self.is_active:
            return []
        
        try:
            buffers = [
                self.canal0_buffer,
                self.beamformed_buffer,
                self.beamformed_filtrado_buffer
            ]
            
            tiempo = np.linspace(0, self.buffer_duration, self.buffer_size)
            
            for col in range(3):
                buffer_actual = buffers[col]
                
                self.lineas_temporales[col].set_data(tiempo, buffer_actual)
                
                if len(buffer_actual) >= 256:
                    f, t, Sxx = signal.spectrogram(buffer_actual, self.sample_rate, 
                                                  nperseg=256, noverlap=128)
                    if Sxx.size > 0:
                        self.imagenes_espectrograma[col].set_data(Sxx)
                        self.imagenes_espectrograma[col].set_extent([0, self.buffer_duration, f[0], f[-1]])
                        self.imagenes_espectrograma[col].set_clim(
                            vmin=np.percentile(Sxx, 10), 
                            vmax=np.percentile(Sxx, 90)
                        )
                
                fft_signal = np.fft.rfft(buffer_actual * np.hanning(len(buffer_actual)))
                fft_magnitude = 20 * np.log10(np.abs(fft_signal) + 1e-8)
                frecuencias = np.fft.rfftfreq(len(buffer_actual), 1/self.sample_rate)
                
                self.lineas_espectro[col].set_data(frecuencias, fft_magnitude)
                self.axes[2, col].set_ylim(
                    np.min(fft_magnitude) - 5, 
                    np.max(fft_magnitude) + 5
                )
            
            todos_elementos = (
                list(self.lineas_temporales) + 
                list(self.imagenes_espectrograma) + 
                list(self.lineas_espectro)
            )
            return todos_elementos
                    
        except Exception as e:
            print(f"❌ Error actualizando gráfica: {e}")
            return []

    def iniciar_beamforming(self):
        """Inicia el beamforming"""
        if self.is_active:
            return
            
        self.is_active = True
        self.is_processing = True
        self.buffer_count = 0
        self.compression_state = 1.0
        self.consecutive_stable_frames = 0
        
        print("✅ BEAMFORMING OPTIMIZADO ACTIVADO")
        print("   - Adaptado a DOA ultra estable")
        print("   - Ganancia adaptativa por confianza")
        print("   - Procesamiento mínimo (máxima estabilidad)")

    def detener_beamforming(self):
        """Detiene el beamforming"""
        if len(self.full_beamformed_audio) > 0:
            self.guardar_audio_beamformed()
        
        self.is_processing = False
        self.is_active = False
        self.buffer_count = 0
        print("🛑 Beamforming detenido")