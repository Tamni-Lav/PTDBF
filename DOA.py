"""
DOA.py - Direction of Arrival SRP-PHAT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

class DOA:
    def __init__(self, gestion_audio):
        self.gestion_audio = gestion_audio
        
        # Configuracion geometrica del array
        self.radio = 0.0325
        self.posiciones = np.array([
            [-self.radio, 0],    # Microfono 1 (canal 2)
            [0, -self.radio],    # Microfono 2 (canal 3)  
            [self.radio, 0],     # Microfono 3 (canal 4)
            [0, self.radio]      # Microfono 4 (canal 5)
        ])
        
        # Parametros de audio
        self.sample_rate = 16000
        self.blocksize = 1024
        self.sound_speed = 343.0
        
        # Estado del sistema
        self.angulo_actual = 0
        self.confianza = 0
        self.is_active = False
        
        # Optimizaciones DOA
        self.resolucion_grados = 1
        self.angulos = np.deg2rad(np.arange(0, 360, self.resolucion_grados))
        self.pares_mic = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
        
        # Busqueda en dos fases
        self.fase_gruesa = False
        self.angulo_grueso = 0
        
        # Pre-calculos
        self.delays = self.delays_precalculados()
        self.precalcular_frecuencias()
        
        # Registrarse para recibir audio
        self.gestion_audio.agregar_suscriptor(self.recibir_audio)
        
        # Configurar grafica
        self.setup_grafica_polar()
        
        print("DOA inicializado - Usando canales 2,3,4,5")

    def recibir_audio(self, audio_data):
        """Callback que recibe audio del gestor central"""
        if self.is_active and audio_data is not None:
            try:
                mic_data = audio_data[:, 1:5].copy()
                self.calcular_doa(mic_data)
            except Exception as e:
                print(f"Error procesando audio en DOA: {e}")

    def precalcular_frecuencias(self):
        """Pre-calcula frecuencias para optimizacion"""
        fft_size = self.blocksize
        frecs = np.fft.fftfreq(fft_size, 1/self.sample_rate)
        frecs_positivas = frecs[:fft_size//2]
        
        self.indices_frecuencias = []
        self.frecuencias_reales = []
        
        salto = 3
        frec_min, frec_max = 300, 3400
        
        for i in range(0, len(frecs_positivas), salto):
            frec = frecs_positivas[i]
            if frec_min <= frec <= frec_max:
                self.indices_frecuencias.append(i)
                self.frecuencias_reales.append(frec)
        
        self.indices_frecuencias = np.array(self.indices_frecuencias)
        self.frecuencias_reales = np.array(self.frecuencias_reales)

    def delays_precalculados(self):
        """Pre-calcula delays teoricos"""
        delays = np.zeros((len(self.angulos), 4))
        
        for i, angulo in enumerate(self.angulos):
            direccion = np.array([np.cos(angulo), np.sin(angulo)])
            for mic in range(4):
                distancia = np.dot(self.posiciones[mic], direccion)
                delays[i, mic] = distancia / self.sound_speed
                
        return delays

    def calcular_potencia_angulo(self, fft_positiva, angle_idx):
        """Calcula potencia SRP-PHAT para un angulo especifico"""
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
        """Algoritmo SRP-PHAT optimizado con busqueda en dos fases"""
        # Aplicar ventana
        ventana = np.hanning(len(audio_frame))
        audio_ventaneado = audio_frame * ventana[:, np.newaxis]
        
        # Transformada de Fourier
        fft_data = np.fft.fft(audio_ventaneado, axis=0)
        fft_positiva = fft_data[:len(fft_data)//2, :]
        
        # Busqueda en dos fases
        if not self.fase_gruesa:
            # Fase 1: Busqueda gruesa rapida (cada 10°)
            angulos_grueso = np.arange(0, 360, 10)
            pots_grueso = np.zeros(len(angulos_grueso))
            
            for i, angulo_grados in enumerate(angulos_grueso):
                angle_idx = int(angulo_grados)
                pots_grueso[i] = self.calcular_potencia_angulo(fft_positiva, angle_idx)
            
            max_idx_grueso = np.argmax(pots_grueso)
            self.angulo_grueso = angulos_grueso[max_idx_grueso]
            self.fase_gruesa = True
            
            if np.max(pots_grueso) < 1e-10:
                self.fase_gruesa = False
                return self.angulo_actual, self.confianza * 0.8
        
        # Fase 2: Busqueda fina (±15°)
        angulo_inicio = max(0, self.angulo_grueso - 15)
        angulo_fin = min(360, self.angulo_grueso + 15)
        indices_busqueda = np.arange(angulo_inicio, angulo_fin, 1)
        
        pots = np.zeros(len(indices_busqueda))
        
        for i, angle_idx in enumerate(indices_busqueda):
            pots[i] = self.calcular_potencia_angulo(fft_positiva, int(angle_idx))
        
        # Estimacion final
        if np.max(pots) > 1e-12:
            max_idx_local = np.argmax(pots)
            angulo_estimado = indices_busqueda[max_idx_local]
            confianza = (pots[max_idx_local] - np.min(pots)) / (np.max(pots) - np.min(pots) + 1e-12)
            
            if np.random.random() < 0.1:
                self.fase_gruesa = False
        else:
            angulo_estimado = self.angulo_actual
            confianza = self.confianza * 0.7
            self.fase_gruesa = False
            
        return angulo_estimado, confianza

    def calcular_doa(self, audio_frame):
        """Procesa un frame de audio y actualiza la estimacion DOA"""
        try:
            audio_filtrado = audio_frame - np.mean(audio_frame, axis=0)
            rms = np.sqrt(np.mean(audio_filtrado**2))
            
            # Deteccion de actividad de voz
            if rms > 0.005:
                angle, confianza = self.srp_phat_optimizado(audio_filtrado)
                
                if confianza > 0.3:
                    self.angulo_actual = angle
                    self.confianza = confianza
                else:
                    self.confianza *= 0.8
            else:
                self.confianza = 0
                self.fase_gruesa = False
                
        except Exception as e:
            print(f"Error en calculo DOA: {e}")

    def setup_grafica_polar(self):
        """Configura la grafica polar"""
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
        
        self.ax.set_title('DOA SRP-PHAT\n', fontsize=16, fontweight='bold', y=1)
        
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

        # Boton de paro
        self.boton_paro_ax = plt.axes([0.82, 0.02, 0.15, 0.05])
        self.boton_paro = Button(self.boton_paro_ax, 'PARAR', color='lightcoral', hovercolor='red')
        self.boton_paro.on_clicked(self.detener_doa)

    def update_plot(self, frame):
        """Actualiza la grafica en tiempo real"""
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
        
        self.angle_text.set_text(f'Direccion: {self.angulo_actual:.1f}°')
        self.conf_text.set_text(f'Confianza: {self.confianza:.2f}')
        
        return [self.angle_line, self.arrow_head, self.angle_text, self.conf_text]

    def iniciar_doa(self):
        """Inicia el sistema DOA"""
        self.is_active = True
        print("DOA ACTIVADO")

    def detener_doa(self, event=None):
        """Detiene el sistema DOA"""
        self.is_active = False
        print("DOA DETENIDO")

    def get_angulo_actual(self):
        """Devuelve el angulo actual detectado"""
        return self.angulo_actual, self.confianza