"""
DOA.py - Direction of Arrival SRP-PHAT CORREGIDO
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from scipy import signal

class DOA:
    def __init__(self, gestion_audio):
        self.gestion_audio = gestion_audio
        
        # Configuración geométrica
        self.radio = 0.0325
        self.posiciones = np.array([
            [0, -self.radio],    # Mic 1 - Canal 2
            [self.radio, 0],    # Mic 2 - Canal 3  
            [0, self.radio],     # Mic 3 - Canal 4
            [-self.radio, 0]      # Mic 4 - Canal 5
        ])
        
        # Parámetros de audio
        self.sample_rate = 16000
        self.blocksize = 1024
        self.sound_speed = 343.0
        
        # Estado del sistema
        self.angulo_actual = 0
        self.confianza = 0
        self.is_active = False
        self.offset_calibracion = 0
        
        # Configuración simplificada para estabilidad
        self.resolucion_grados = 5  # Más rápido, menos preciso inicialmente
        self.angulos = np.arange(0, 360, self.resolucion_grados)
        self.angulos_rad = np.deg2rad(self.angulos)
        
        # Solo pares adyacentes para velocidad
        self.pares_mic = [(0, 1), (1, 2), (2, 3), (3, 0)]
        
        # Rango de frecuencias conservador
        self.frec_min = 500
        self.frec_max = 3000
        
        # Buffers para suavizado
        self.buffer_angulos = []
        self.buffer_size = 8
        
        # Pre-cálculos
        self.delays = self.delays_precalculados()
        self.precalcular_frecuencias()
        
        # Estado de búsqueda
        self.fase_gruesa = False
        self.angulo_grueso = 0
        
        # Registrarse para recibir audio
        self.gestion_audio.agregar_suscriptor(self.recibir_audio)
        
        # Configurar gráfica
        self.setup_grafica_polar()
        
        print("DOA inicializado - MODO ESTABLE")

    def recibir_audio(self, audio_data):
        """Callback que recibe audio del gestor central"""
        if self.is_active and audio_data is not None:
            try:
                # Usar canales 2,3,4,5 (índices 1,2,3,4)
                if audio_data.shape[1] >= 5:
                    mic_data = audio_data[:, 1:5].copy()
                    self.calcular_doa(mic_data)
            except Exception as e:
                print(f"Error en DOA recibir_audio: {e}")

    def precalcular_frecuencias(self):
        """Pre-calcula frecuencias de manera segura"""
        fft_size = self.blocksize
        frecs = np.fft.fftfreq(fft_size, 1/self.sample_rate)
        frecs_positivas = frecs[:fft_size//2]
        
        self.indices_frecuencias = []
        self.frecuencias_reales = []
        
        # Procesar cada frecuencia en el rango (más simple)
        for i, frec in enumerate(frecs_positivas):
            if self.frec_min <= frec <= self.frec_max:
                self.indices_frecuencias.append(i)
                self.frecuencias_reales.append(frec)
        
        self.indices_frecuencias = np.array(self.indices_frecuencias)
        self.frecuencias_reales = np.array(self.frecuencias_reales)
        
        print(f"DOA: {len(self.indices_frecuencias)} frecuencias en {self.frec_min}-{self.frec_max}Hz")

    def delays_precalculados(self):
        """Pre-calcula delays teóricos"""
        delays = np.zeros((len(self.angulos), 4))
        
        for i, angulo_deg in enumerate(self.angulos):
            angulo_rad = np.deg2rad(angulo_deg)
            direccion = np.array([np.cos(angulo_rad), np.sin(angulo_rad)])
            for mic in range(4):
                distancia = np.dot(self.posiciones[mic], direccion)
                delays[i, mic] = distancia / self.sound_speed
                
        return delays

    def calcular_potencia_angulo(self, fft_positiva, angle_idx):
        """Calcula potencia SRP-PHAT para un ángulo"""
        pot = 0.0
        
        # Asegurar que el índice esté en rango
        if angle_idx >= len(self.angulos):
            angle_idx = angle_idx % len(self.angulos)
        
        for mic1, mic2 in self.pares_mic:
            tau = self.delays[angle_idx, mic1] - self.delays[angle_idx, mic2]
            
            # Verificar límites del array
            valid_indices = self.indices_frecuencias[self.indices_frecuencias < fft_positiva.shape[0]]
            if len(valid_indices) == 0:
                continue
                
            frecs_validas = self.frecuencias_reales[self.indices_frecuencias < fft_positiva.shape[0]]
            fases_esperadas = 2 * np.pi * frecs_validas * tau
            
            try:
                X1 = fft_positiva[valid_indices, mic1]
                X2 = fft_positiva[valid_indices, mic2]
                
                cross = X1 * np.conj(X2)
                magnitudes = np.abs(cross)
                
                mask = magnitudes > 1e-12
                if np.any(mask):
                    cross_phat = cross[mask] / magnitudes[mask]
                    pot += np.sum(np.real(cross_phat * np.exp(1j * fases_esperadas[mask])))
            except IndexError as e:
                continue
        
        return pot

    def srp_phat_estable(self, audio_frame):
        """Algoritmo SRP-PHAT estabilizado"""
        try:
            # Aplicar ventana
            ventana = np.hanning(len(audio_frame))
            audio_ventaneado = audio_frame * ventana[:, np.newaxis]
            
            # FFT segura
            fft_data = np.fft.fft(audio_ventaneado, axis=0)
            fft_positiva = fft_data[:len(fft_data)//2, :]
            
            # Búsqueda simple en todos los ángulos
            pots = np.zeros(len(self.angulos))
            
            for i, angulo_grados in enumerate(self.angulos):
                pots[i] = self.calcular_potencia_angulo(fft_positiva, i)
            
            if np.max(pots) > 1e-12:
                max_idx = np.argmax(pots)
                angulo_estimado = self.angulos[max_idx]
                
                # Confianza simple
                confianza = (pots[max_idx] - np.min(pots)) / (np.max(pots) - np.min(pots) + 1e-12)
                confianza = min(1.0, max(0.0, confianza))
            else:
                angulo_estimado = self.angulo_actual
                confianza = 0.0
            
            return int(angulo_estimado), confianza
            
        except Exception as e:
            print(f"Error en SRP-PHAT: {e}")
            return self.angulo_actual, 0.0

    def aplicar_suavizado(self, nuevo_angulo, nueva_confianza):
        """Suavizado simple y efectivo"""
        if nueva_confianza < 0.3:
            # Baja confianza, mantener ángulo anterior
            return self.angulo_actual, nueva_confianza * 0.5
        
        # Agregar al buffer
        self.buffer_angulos.append(nuevo_angulo)
        if len(self.buffer_angulos) > self.buffer_size:
            self.buffer_angulos.pop(0)
        
        if len(self.buffer_angulos) < 3:
            return nuevo_angulo, nueva_confianza
        
        # Usar mediana del buffer
        angulo_suavizado = int(np.median(self.buffer_angulos))
        
        # Aplicar calibración
        angulo_calibrado = (angulo_suavizado + self.offset_calibracion) % 360
        
        return angulo_calibrado, nueva_confianza

    def calcular_doa(self, audio_frame):
        """Procesa un frame de audio de manera estable"""
        try:
            # Pre-procesamiento básico
            audio_filtrado = audio_frame - np.mean(audio_frame, axis=0)
            rms = np.sqrt(np.mean(audio_filtrado**2))
            
            # Detección de actividad
            if rms > 0.005:  # Umbral de voz
                angle, confianza = self.srp_phat_estable(audio_filtrado)
                
                # Aplicar suavizado
                angle_suavizado, confianza_suavizada = self.aplicar_suavizado(angle, confianza)
                
                if confianza_suavizada > 0.2:
                    self.angulo_actual = angle_suavizado
                    self.confianza = confianza_suavizada
                else:
                    self.confianza *= 0.9
            else:
                # Sin actividad, reducir confianza gradualmente
                self.confianza = max(0, self.confianza - 0.1)
                
        except Exception as e:
            #print(f"Error en cálculo DOA: {e}")  # Comentado para reducir spam
            pass

    def setup_grafica_polar(self):
        """Configura la gráfica polar simplificada"""
        try:
            self.fig, self.ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            self.ax.set_theta_zero_location('E')
            self.ax.set_theta_direction(1)
            self.ax.set_ylim(0, 1.2)
            self.ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            self.ax.set_yticklabels(['', '', '', '', ''])
            self.ax.grid(True, alpha=0.7)
            
            marcas_angulos = ['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°']
            angulos_rad = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])
            self.ax.set_xticks(angulos_rad)
            self.ax.set_xticklabels(marcas_angulos)
            
            self.ax.set_title('DOA - Localización de Fuente Sonora\n', fontsize=16, fontweight='bold', y=1)
            
            self.angle_line, = self.ax.plot([], [], 'r-', linewidth=6, alpha=0.8)
            
            self.arrow_head = self.ax.annotate('', 
                                              xy=(0, 0),
                                              xytext=(0, 0),
                                              arrowprops=dict(arrowstyle='->', 
                                                             color='red', 
                                                             lw=3,
                                                             alpha=0.8,
                                                             mutation_scale=15),
                                              annotation_clip=False)
            
            self.angle_text = self.ax.text(0.5, 0.15, 'Inicializando...', transform=self.ax.transAxes, 
                                          ha='center', fontsize=20, fontweight='bold',
                                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            self.conf_text = self.ax.text(0.5, 0.05, 'Confianza: 0.00', transform=self.ax.transAxes,
                                         ha='center', fontsize=12, color='green')

            # Botón de paro
            self.boton_paro_ax = plt.axes([0.82, 0.02, 0.15, 0.05])
            self.boton_paro = Button(self.boton_paro_ax, 'PARAR', color='lightcoral', hovercolor='red')
            self.boton_paro.on_clicked(self.detener_doa)
            
        except Exception as e:
            print(f"Error configurando gráfica DOA: {e}")

    def update_plot(self, frame):
        """Actualiza la gráfica de manera segura"""
        try:
            current_rad = np.deg2rad(self.angulo_actual)
            
            self.angle_line.set_data([current_rad, current_rad], [0, 0.9])
            
            arrow_x = current_rad
            arrow_y = 0.9
            
            self.arrow_head.xy = (arrow_x, arrow_y)
            
            # Color según confianza
            if self.confianza > 0.7:
                color = 'green'
            elif self.confianza > 0.3:
                color = 'orange'
            else:
                color = 'red'
                
            self.angle_line.set_color(color)
            self.arrow_head.arrow_patch.set_color(color)
            
            self.angle_text.set_text(f'Dirección: {self.angulo_actual}°')
            self.conf_text.set_text(f'Confianza: {self.confianza:.2f}')
            
            return [self.angle_line, self.arrow_head, self.angle_text, self.conf_text]
            
        except Exception as e:
            return []

    def iniciar_doa(self):
        """Inicia el sistema DOA"""
        self.is_active = True
        self.buffer_angulos = []
        print("DOA ACTIVADO")

    def detener_doa(self, event=None):
        """Detiene el sistema DOA"""
        self.is_active = False
        print("DOA DETENIDO")

    def get_angulo_actual(self):
        """Devuelve el ángulo actual detectado"""
        return self.angulo_actual, self.confianza

    def set_calibracion(self, offset):
        """Establece el offset de calibración"""
        self.offset_calibracion = offset
        print(f"DOA: Offset de calibración establecido a {offset}°")