"""
SistemaIntegrado.py - Sistema principal integrado DOA + Beamforming.
15251
"""

import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from GestionDispositivos import GestionDispositivos
from PDG import MicrophoneArrayRealtime
from DOA import DOA
from Beamforming import BeamformingSystem


class SistemaIntegrado:
    """Sistema principal que integra todos los módulos de procesamiento."""
    
    def __init__(self):
        """Inicializa el sistema integrado completo."""
        print("=" * 60)
        print("SISTEMA INTEGRADO - DOA + BEAMFORMING")
        print("=" * 60)
        
        # Inicializar gestión de audio
        self.gestion_audio = GestionDispositivos(channels=6)
        
        if self.gestion_audio.device_index is None:
            print("Error: No se pudo inicializar el audio")
            return
        
        # Inicializar módulos
        self.pdg = MicrophoneArrayRealtime(self.gestion_audio)
        self.doa = DOA(self.gestion_audio)
        self.beamforming = BeamformingSystem(self.gestion_audio, self.doa)
        
        self.animaciones = []
        self.sistema_activo = False

    def iniciar_sistema(self):
        """Inicia todo el sistema integrado.
        
        Returns:
            bool: True si se inició correctamente
        """
        try:
            print("\nIniciando sistema completo...")
            
            # 1. Iniciar captura de audio
            print("1. Iniciando captura de audio...")
            if not self.gestion_audio.iniciar_captura():
                print("Error: Falló la captura de audio")
                return False
            
            time.sleep(0.5)
            
            # 2. Iniciar procesamiento de módulos
            print("2. Iniciando módulos de procesamiento...")
            self.pdg.running = True
            self.doa.iniciar_doa()
            self.beamforming.iniciar_beamforming()
            
            # 3. Configurar visualizaciones
            print("3. Configurando visualizaciones...")
            self.configurar_visualizaciones()
            
            self.sistema_activo = True
            
            print("\nSistema iniciado correctamente")
            print(" - PDG: Visualización de micrófonos")
            print(" - DOA: Localización de fuente sonora") 
            print(" - Beamforming: Procesamiento direccional")
            
            plt.show()
            return True
            
        except Exception as e:
            print(f"Error iniciando sistema: {e}")
            import traceback
            traceback.print_exc()
            return False

    def configurar_visualizaciones(self):
        """Configura todas las visualizaciones del sistema."""
        # PDG - Visualización de micrófonos
        self.pdg.setup_graficos()
        ani_pdg = FuncAnimation(
            self.pdg.fig, 
            self.pdg.update_plot, 
            interval=33, 
            blit=False, 
            cache_frame_data=False
        )
        self.animaciones.append(ani_pdg)
        print("   PDG configurado")
        
        # DOA - Localización
        ani_doa = FuncAnimation(
            self.doa.fig, 
            self.doa.update_plot,
            interval=50, 
            blit=False, 
            cache_frame_data=False
        )
        self.animaciones.append(ani_doa)
        print("   DOA configurado")
        
        # Beamforming - Procesamiento direccional
        self.configurar_beamforming()
        print("   Beamforming configurado")

    def configurar_beamforming(self):
        """Configura la visualización del beamforming."""
        try:
            self.fig_bf = plt.figure(figsize=(12, 8))
            
            if self.beamforming.configurar_visualizacion(self.fig_bf):
                ani_bf = FuncAnimation(
                    self.fig_bf, 
                    self.beamforming.update_plot,
                    interval=100, 
                    blit=False, 
                    cache_frame_data=False
                )
                self.animaciones.append(ani_bf)
            
        except Exception as e:
            print(f"Visualización beamforming no disponible: {e}")

    def detener_sistema(self):
        """Detiene el sistema completo."""
        print("\nDeteniendo sistema...")
        self.sistema_activo = False
        
        if hasattr(self, 'beamforming'):
            self.beamforming.detener_beamforming()
        if hasattr(self, 'doa'):
            self.doa.detener_doa()
        if hasattr(self, 'pdg'):
            self.pdg.running = False
        if hasattr(self, 'gestion_audio'):
            self.gestion_audio.detener_captura()
        
        for ani in self.animaciones:
            try:
                ani.event_source.stop()
            except:
                pass
        
        plt.close('all')
        print("Sistema detenido")


def main():
    """Función principal del sistema."""
    sistema = SistemaIntegrado()
    
    if not hasattr(sistema, 'gestion_audio') or sistema.gestion_audio.device_index is None:
        print("Error: No se pudo inicializar el sistema de audio")
        return
    
    try:
        sistema.iniciar_sistema()
    except KeyboardInterrupt:
        print("\nSistema interrumpido por el usuario")
    except Exception as e:
        print(f"Error inesperado: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sistema.detener_sistema()


if __name__ == "__main__":
    main()