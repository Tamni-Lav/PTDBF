"""
DOA.py - Direction of Arrival usando SRP-PHAT optimizado.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button


class DOA:
    """Sistema de estimación de dirección de llegada usando SRP-PHAT optimizado."""
    
    def __init__(self, gestion_audio):
        """Inicializa el sistema DOA.
        
        Args:
            gestion_audio: Gestor central de audio
        """
        self.gestion_audio = gestion_audio
        
        # Configuración geométrica del array
        self.radio = 0.0325
        self.posiciones = np.array([
            [-self.radio, 0],    # Micrófono 1 (canal 2)
            [0, -self.radio],    # Micrófono 2 (canal 3)  
            [self.radio, 0],     # Micrófono 3 (canal 4)
            [0, self.radio]      # Micrófono 4 (canal 5)
        ])
        
        # Parámetros de audio
        self.sample_rate = 16000
        self.blocksize = 1024
        self.sound_speed = 343.0
        
        # Estado del sistema
        self.angulo_actual = 0
        self.confianza = 0
        self.is_active = False
        
        # OPTIMIZACIÓN 1: Mejor resolución angular
        self.resolucion_grados = 0.25  # Mejorado de 1° a 0.25°
        self.angulos = np.deg2rad(np.arange(0, 360, self.resolucion_grados))
        self.pares_mic = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
        
        # Búsqueda en dos fases
        self.fase_gruesa = False
        self.angulo_grueso = 0
        
        # Pre-cálculos
        self.delays = self.delays_precalculados()
        
        # OPTIMIZACIÓN 2: Frecuencias estratégicas para voz
        self.precalcular_frecuencias_optimizadas()
        
        # Registrarse para recibir audio
        self.gestion_audio.agregar_suscriptor(self.recibir_audio)
        
        # Configurar gráfica
        self.setup_grafica_polar()
        
        print("DOA inicializado - Resolución 0.25° - Frecuencias optimizadas")

    def recibir_audio(self, audio_data):
        """Callback que recibe audio del gestor central.
        
        Args:
            audio_data: Datos de audio capturados
        """
        if self.is_active and audio_data is not None:
            try:
                mic_data = audio_data[:, 1:5].copy()
                self.calcular_doa(mic_data)
            except Exception as e:
                print(f"Error procesando audio en DOA: {e}")

    def precalcular_frecuencias_optimizadas(self):
        """Pre-calcula frecuencias estratégicas optimizadas para voz humana.
        
        OPTIMIZACIÓN: 11 frecuencias clave en formantes vocales
        Menos cálculos que el método anterior pero mejor distribución
        """
        # Frecuencias estratégicas para voz humana (Hz)
        frecuencias_strategicas = [
            350,   # F1 - vocales graves
            500,   # F1 - energía fundamental
            700,   # F1-F2 transición
            900,   # F2 - inicio formantes medios
            1200,  # F2 - formantes principales (IMPORTANTE)
            1500,  # F2 - máximo energía vocal
            1800,  # F2-F3 transición
            2200,  # F3 - claridad e inteligibilidad
            2600,  # F3 - detalles vocales
            3000,  # F3-F4 - límite superior útil
            3300   # F4 - componentes de alta frecuencia
        ]
        
        # Convertir a índices FFT
        fft_size = self.blocksize
        frecs = np.fft.fftfreq(fft_size, 1/self.sample_rate)
        frecs_positivas = frecs[:fft_size//2]
        
        self.indices_frecuencias = []
        self.frecuencias_reales = []
        
        # Encontrar los índices más cercanos a nuestras frecuencias estratégicas
        for frec_objetivo in frecuencias_strategicas:
            idx = np.argmin(np.abs(frecs_positivas - frec_objetivo))
            if 0 <= idx < len(frecs_positivas):
                self.indices_frecuencias.append(idx)
                self.frecuencias_reales.append(frecs_positivas[idx])
        
        self.indices_frecuencias = np.array(self.indices_frecuencias)
        self.frecuencias_reales = np.array(self.frecuencias_reales)
        
        print(f"Frecuencias optimizadas: {len(self.frecuencias_reales)} puntos")
        print(f"Frecuencias: {self.frecuencias_reales.round(1)} Hz")

    def delays_precalculados(self):
        """Pre-calcula delays teóricos para todos los ángulos.
        
        Returns:
            numpy.ndarray: Matriz de delays para cada ángulo y micrófono
        """
        delays = np.zeros((len(self.angulos), 4))
        
        for i, angulo in enumerate(self.angulos):
            direccion = np.array([np.cos(angulo), np.sin(angulo)])
            for mic in range(4):
                distancia = np.dot(self.posiciones[mic], direccion)
                delays[i, mic] = distancia / self.sound_speed
                
        return delays

    def calcular_potencia_angulo(self, fft_positiva, angle_idx):
        """Calcula potencia SRP-PHAT para un ángulo específico.
        
        Args:
            fft_positiva: Transformada de Fourier positiva
            angle_idx: Índice del ángulo a evaluar
            
        Returns:
            float: Potencia calculada para el ángulo
        """
        pot = 0.0
        
        for mic1, mic2 in self.pares_mic:
            tau = self.delays[angle_idx, mic1] - self.delays[angle_idx, mic2]
            
            for j, frec_idx in enumerate(self.indices_frecuencias):
                frec = self.frecuencias_reales[j]
                fase_esperada = 2 * np.pi * frec * tau
                
                X1 = fft_positiva[frec_idx, mic1]
                X2 = fft_positiva[frec_idx, mic2]
                
                cross = X1 * np.conj(X2)
                magnitude = np.abs(cross)
                
                if magnitude > 1e-12:
                    cross_phat = cross / magnitude
                    pot += np.real(cross_phat * np.exp(1j * fase_esperada))
        
        return pot

    def srp_phat_optimizado(self, audio_frame):
        """Algoritmo SRP-PHAT optimizado con búsqueda en dos fases.
        
        Args:
            audio_frame: Frame de audio a procesar
            
        Returns:
            tuple: (ángulo_estimado, confianza)
        """
        # Aplicar ventana
        ventana = np.hanning(len(audio_frame))
        audio_ventaneado = audio_frame * ventana[:, np.newaxis]
        
        # Transformada de Fourier
        fft_data = np.fft.fft(audio_ventaneado, axis=0)
        fft_positiva = fft_data[:len(fft_data)//2, :]
        
        # Búsqueda en dos fases
        if not self.fase_gruesa:
            # Fase 1: Búsqueda gruesa rápida (cada 5° - más preciso)
            angulos_grueso = np.arange(0, 360, 5)
            pots_grueso = np.zeros(len(angulos_grueso))
            
            for i, angulo_grados in enumerate(angulos_grueso):
                angle_idx = int(angulo_grados / self.resolucion_grados)
                pots_grueso[i] = self.calcular_potencia_angulo(fft_positiva, angle_idx)
            
            max_idx_grueso = np.argmax(pots_grueso)
            self.angulo_grueso = angulos_grueso[max_idx_grueso]
            self.fase_gruesa = True
            
            if np.max(pots_grueso) < 1e-10:
                self.fase_gruesa = False
                return self.angulo_actual, self.confianza * 0.8
        
        # Fase 2: Búsqueda fina (±10° - más preciso)
        angulo_inicio = max(0, self.angulo_grueso - 10)
        angulo_fin = min(360, self.angulo_grueso + 10)
        indices_busqueda = np.arange(angulo_inicio, angulo_fin, self.resolucion_grados)
        
        pots = np.zeros(len(indices_busqueda))
        
        for i, angle_deg in enumerate(indices_busqueda):
            angle_idx = int(angle_deg / self.resolucion_grados)
            pots[i] = self.calcular_potencia_angulo(fft_positiva, angle_idx)
        
        # Estimación final
        if np.max(pots) > 1e-12:
            max_idx_local = np.argmax(pots)
            angulo_estimado = indices_busqueda[max_idx_local]
            
            # Mejor cálculo de confianza
            pots_ordenadas = np.sort(pots)
            confianza = (pots[max_idx_local] - np.median(pots)) / (np.max(pots) - np.min(pots) + 1e-12)
            confianza = np.clip(confianza, 0, 1)
            
            # Reiniciar búsqueda gruesa ocasionalmente
            if np.random.random() < 0.05:
                self.fase_gruesa = False
        else:
            angulo_estimado = self.angulo_actual
            confianza = self.confianza * 0.7
            self.fase_gruesa = False
            
        return angulo_estimado, confianza

    def calcular_doa(self, audio_frame):
        """Procesa un frame de audio y actualiza la estimación DOA.
        
        Args:
            audio_frame: Frame de audio a procesar
        """
        try:
            audio_filtrado = audio_frame - np.mean(audio_frame, axis=0)
            rms = np.sqrt(np.mean(audio_filtrado**2))
            
            # Detección de actividad de voz mejorada
            if rms > 0.004:  # Ligeramente más sensible
                angle, confianza = self.srp_phat_optimizado(audio_filtrado)
                
                if confianza > 0.25:  # Umbral más bajo para mejor respuesta
                    self.angulo_actual = angle
                    self.confianza = confianza
                else:
                    self.confianza *= 0.9  # Decaimiento más lento
            else:
                self.confianza = 0
                self.fase_gruesa = False
                
        except Exception as e:
            print(f"Error en cálculo DOA: {e}")

    def setup_grafica_polar(self):
        """Configura la gráfica polar para visualización."""
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
        
        self.ax.set_title('DOA SRP-PHAT Optimizado\n', fontsize=16, fontweight='bold', y=1)
        
        self.angle_line, = self.ax.plot([], [], 'r-', linewidth=6, alpha=0)
        
        self.arrow_head = self.ax.annotate('', 
                                          xy=(0, 0),
                                          xytext=(0, 0),
                                          arrowprops=dict(arrowstyle='->', 
                                                         color='red', 
                                                         lw=3,
                                                         alpha=0.8,
                                                         mutation_scale=15),
                                          annotation_clip=False)
        
        self.angle_text = self.ax.text(0.5, 0.15, '', transform=self.ax.transAxes, 
                                      ha='center', fontsize=20, fontweight='bold',
                                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        self.conf_text = self.ax.text(0.5, 0.05, '', transform=self.ax.transAxes,
                                     ha='center', fontsize=12, color='green')

        # Botón de paro
        self.boton_paro_ax = plt.axes([0.82, 0.02, 0.15, 0.05])
        self.boton_paro = Button(self.boton_paro_ax, 'PARAR', color='lightcoral', hovercolor='red')
        self.boton_paro.on_clicked(self.detener_doa)

    def update_plot(self, frame):
        """Actualiza la gráfica en tiempo real.
        
        Returns:
            list: Elementos actualizados de la gráfica
        """
        current_rad = np.deg2rad(self.angulo_actual)
        
        self.angle_line.set_data([current_rad, current_rad], [0, 0.9])
        
        arrow_x = current_rad
        arrow_y = 0.9
        
        self.arrow_head.xy = (arrow_x, arrow_y)
        
        if self.confianza > 0.7:
            color = 'green'
        elif self.confianza > 0.3:
            color = 'orange'
        else:
            color = 'red'
            
        self.angle_line.set_color(color)
        self.arrow_head.arrow_patch.set_color(color)
        
        self.angle_text.set_text(f'Dirección: {self.angulo_actual:.1f}°')
        self.conf_text.set_text(f'Confianza: {self.confianza:.2f}')
        
        return [self.angle_line, self.arrow_head, self.angle_text, self.conf_text]

    def iniciar_doa(self):
        """Inicia el sistema DOA."""
        self.is_active = True
        print("DOA ACTIVADO - Resolución 0.25° - Frecuencias optimizadas")

    def detener_doa(self, event=None):
        """Detiene el sistema DOA."""
        self.is_active = False
        print("DOA DETENIDO")

    def get_angulo_actual(self):
        """Devuelve el ángulo actual detectado.
        
        Returns:
            tuple: (ángulo, confianza)
        """
        return self.angulo_actual, self.confianza