"""
SISTEMA INTEGRADO - Versión Estable y Corregida
vcaa - SIN ERRORES DE ÍNDICE
"""

import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from GestionDispositivos import GestionDispositivos
from PDG import MicrophoneArrayRealtime
from DOA import DOA
from Beamforming import BeamformingSystem
from CalibracionDOA import CalibradorDOA

class SistemaIntegrado:
    def __init__(self):
        print("=" * 60)
        print("SISTEMA INTEGRADO - DOA + BEAMFORMING ESTABLE")
        print("=" * 60)
        
        # Inicializar gestión de audio
        self.gestion_audio = GestionDispositivos(channels=6)
        
        if self.gestion_audio.device_index is None:
            print("❌ Error: No se pudo inicializar el audio")
            return
        
        # Inicializar módulos
        self.pdg = MicrophoneArrayRealtime(self.gestion_audio)
        self.doa = DOA(self.gestion_audio)
        self.beamforming = BeamformingSystem(self.gestion_audio, self.doa)
        
        # Inicializar calibrador
        self.calibrador = CalibradorDOA(self.doa)
        
        self.animaciones = []
        self.sistema_activo = False

    def iniciar_sistema(self):
        """Inicia todo el sistema integrado de manera estable"""
        try:
            print("\nIniciando sistema completo...")
            
            # 1. Iniciar captura de audio
            print("1. Iniciando captura de audio...")
            if not self.gestion_audio.iniciar_captura():    
                print("❌ Error: Falló la captura de audio")
                return False
            
            time.sleep(1.0)  # Tiempo para estabilizar el audio
            
            # 2. Pregunta simple de calibración
            print("\n" + "="*40)
            print("🎯 CALIBRACIÓN DOA")
            print("="*40)
            print("¿Deseas calibrar el sistema?")
            print("• Recomendado para primera vez")
            print("• Coloca fuente a 0° y habla/música")
            print("• Duración: 8 segundos")
            
            respuesta = input("\nCalibrar? (s/n): ").lower().strip()
            
            if respuesta == 's':
                print("\n🔄 Iniciando calibración rápida...")
                resultado = self.calibrador.calibrar_rapido(0, 8)
                if resultado is not None:
                    print("✅ Calibración completada")
                else:
                    print("⚠️  Calibración no completada - continuando igual")
            
            # 3. Iniciar módulos de procesamiento
            print("\n2. Iniciando módulos de procesamiento...")
            self.pdg.running = True
            self.doa.iniciar_doa()
            self.beamforming.iniciar_beamforming()
            
            time.sleep(0.5)  # Pequeña pausa para estabilizar
            
            # 4. Configurar visualizaciones
            print("3. Configurando visualizaciones...")
            if not self.configurar_visualizaciones():
                print("⚠️  Algunas visualizaciones no están disponibles")
            
            self.sistema_activo = True
            
            print("\n" + "="*50)
            print("✅ SISTEMA INICIADO CORRECTAMENTE")
            print("="*50)
            print("   - PDG: Visualización de micrófonos")
            print("   - DOA: Localización de fuente sonora") 
            print("   - Beamforming: Procesamiento direccional")
            print("\n💡 Comandos útiles:")
            print("   • sistema.menu_calibracion() - Recalibrar")
            print("   • sistema.detener_sistema() - Apagar")
            print("="*50)
            
            # Mostrar estado inicial
            estado = self.gestion_audio.obtener_estado()
            print(f"\n📊 Estado del sistema:")
            print(f"   - Muestras/segundo: {estado['sample_rate']}")
            print(f"   - Tamaño de bloque: {estado['blocksize']}")
            print(f"   - Canales activos: {estado['channels']}")
            print(f"   - Suscriptores: {estado['suscriptores']}")
            
            plt.show()
            
            return True
            
        except Exception as e:
            print(f"❌ Error iniciando sistema: {e}")
            import traceback
            traceback.print_exc()
            return False

    def configurar_visualizaciones(self):
        """Configura todas las visualizaciones del sistema"""
        try:
            success_count = 0
            
            # PDG - Visualización de micrófonos
            try:
                self.pdg.setup_graficos()
                ani_pdg = FuncAnimation(
                    self.pdg.fig, self.pdg.update_plot, 
                    interval=50, blit=False, cache_frame_data=False
                )
                self.animaciones.append(ani_pdg)
                print("   ✅ PDG configurado")
                success_count += 1
            except Exception as e:
                print(f"   ❌ PDG no disponible: {e}")
            
            # DOA - Localización
            try:
                ani_doa = FuncAnimation(
                    self.doa.fig, self.doa.update_plot,
                    interval=100, blit=False, cache_frame_data=False
                )
                self.animaciones.append(ani_doa)
                print("   ✅ DOA configurado")
                success_count += 1
            except Exception as e:
                print(f"   ❌ DOA no disponible: {e}")
            
            # Beamforming - Procesamiento direccional
            try:
                if self.configurar_beamforming():
                    print("   ✅ Beamforming configurado")
                    success_count += 1
                else:
                    print("   ⚠️  Beamforming limitado")
            except Exception as e:
                print(f"   ❌ Beamforming no disponible: {e}")
            
            return success_count > 0
            
        except Exception as e:
            print(f"❌ Error en visualizaciones: {e}")
            return False

    def configurar_beamforming(self):
        """Configura la visualización del beamforming"""
        try:
            self.fig_bf = plt.figure(figsize=(12, 8))
            
            if hasattr(self.beamforming, 'configurar_visualizacion'):
                if self.beamforming.configurar_visualizacion(self.fig_bf):
                    ani_bf = FuncAnimation(
                        self.fig_bf, self.beamforming.update_plot,
                        interval=150, blit=False, cache_frame_data=False
                    )
                    self.animaciones.append(ani_bf)
                    return True
            return False
            
        except Exception as e:
            print(f"   ⚠️  Visualización beamforming limitada: {e}")
            return False

    def menu_calibracion(self):
        """Menú de calibración del sistema"""
        if not self.sistema_activo:
            print("❌ El sistema no está activo")
            return
        
        print("\n" + "="*50)
        print("🔧 MENÚ DE CALIBRACIÓN")
        print("="*50)
        print("1. Calibración rápida (8 segundos)")
        print("2. Calibración extendida (15 segundos)") 
        print("3. Ver offset actual")
        print("4. Resetear calibración")
        print("5. Volver")
        print("="*50)
        
        try:
            opcion = input("\nSelecciona opción (1-5): ").strip()
            
            if opcion == "1":
                print("\n🎯 Calibración rápida - Habla desde 0°")
                self.calibrador.calibrar_rapido(0, 8)
                
            elif opcion == "2":
                print("\n🎯 Calibración extendida - Habla desde 0°") 
                self.calibrador.calibrar_rapido(0, 15)
                
            elif opcion == "3":
                offset = self.doa.offset_calibracion
                if offset == 0:
                    print("📏 Offset actual: 0° (sin calibración)")
                else:
                    print(f"📏 Offset actual: {offset}°")
                    
            elif opcion == "4":
                self.doa.offset_calibracion = 0
                print("🔄 Calibración reseteada a 0°")
                
            elif opcion == "5":
                return
            else:
                print("❌ Opción inválida")
                
        except KeyboardInterrupt:
            print("\n⚠️ Operación cancelada")
        except Exception as e:
            print(f"❌ Error: {e}")

    def estado_sistema(self):
        """Muestra el estado actual del sistema"""
        if not hasattr(self, 'gestion_audio'):
            print("❌ Sistema no inicializado")
            return
        
        estado_audio = self.gestion_audio.obtener_estado()
        
        print("\n" + "="*50)
        print("📊 ESTADO DEL SISTEMA")
        print("="*50)
        print(f"Audio:")
        print(f"  • Grabación: {'✅ ACTIVA' if estado_audio['recording'] else '❌ INACTIVA'}")
        print(f"  • Dispositivo: {estado_audio['device_index']}")
        print(f"  • Sample rate: {estado_audio['sample_rate']} Hz")
        print(f"  • Blocksize: {estado_audio['blocksize']} muestras")
        print(f"  • Overflows: {estado_audio['overflows']}")
        
        print(f"\nMódulos:")
        print(f"  • PDG: {'✅ ACTIVO' if hasattr(self, 'pdg') and self.pdg.running else '❌ INACTIVO'}")
        print(f"  • DOA: {'✅ ACTIVO' if hasattr(self, 'doa') and self.doa.is_active else '❌ INACTIVO'}")
        print(f"  • Beamforming: {'✅ ACTIVO' if hasattr(self, 'beamforming') and self.beamforming.is_active else '❌ INACTIVO'}")
        
        if hasattr(self, 'doa'):
            angulo, confianza = self.doa.get_angulo_actual()
            print(f"  • Ángulo DOA: {angulo}° (conf: {confianza:.2f})")
            print(f"  • Calibración: {self.doa.offset_calibracion}°")
        
        print(f"\nVisualizaciones: {len(self.animaciones)} activas")
        print("="*50)

    def detener_sistema(self):
        """Detiene el sistema completo de manera segura"""
        print("\n" + "="*50)
        print("🛑 DETENIENDO SISTEMA...")
        print("="*50)
        
        self.sistema_activo = False
        
        # Detener módulos en orden inverso
        if hasattr(self, 'beamforming'):
            print("• Deteniendo beamforming...")
            self.beamforming.detener_beamforming()
            
        if hasattr(self, 'doa'):
            print("• Deteniendo DOA...")
            self.doa.detener_doa()
            
        if hasattr(self, 'pdg'):
            print("• Deteniendo PDG...")
            self.pdg.running = False
            
        if hasattr(self, 'gestion_audio'):
            print("• Deteniendo captura de audio...")
            self.gestion_audio.detener_captura()
        
        # Detener animaciones
        print("• Cerrando visualizaciones...")
        for ani in self.animaciones:
            try:
                ani.event_source.stop()
            except:
                pass
        
        # Cerrar figuras
        try:
            plt.close('all')
        except:
            pass
        
        print("✅ Sistema detenido correctamente")
        print("="*50)

    def __del__(self):
        """Destructor para limpieza segura"""
        if self.sistema_activo:
            self.detener_sistema()

def main():
    """Función principal del sistema"""
    sistema = None
    
    try:
        # Inicializar sistema
        sistema = SistemaIntegrado()
        
        if not hasattr(sistema, 'gestion_audio') or sistema.gestion_audio.device_index is None:
            print("❌ No se pudo inicializar el sistema de audio")
            return
        
        # Iniciar sistema
        if not sistema.iniciar_sistema():
            print("❌ Fallo al iniciar el sistema")
            return
        
        # Mantener el sistema activo
        print("\n💡 Sistema ejecutándose...")
        print("   Presiona Ctrl+C para detener")
        
        # Loop principal simple
        while sistema.sistema_activo:
            try:
                time.sleep(1)
                # Verificar si las ventanas siguen abiertas
                if not plt.get_fignums():
                    print("\n⚠️ Ventanas cerradas - deteniendo sistema...")
                    break
                    
            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupción por usuario")
                break
            except Exception as e:
                print(f"\n⚠️ Error en loop principal: {e}")
                break
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Sistema interrumpido durante inicio")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Limpieza garantizada
        if sistema:
            sistema.detener_sistema()
        print("\n👋 Sistema finalizado")

# Comandos rápidos para depuración
def comandos_rapidos():
    """Función para depuración rápida"""
    print("\n⚡ COMANDOS RÁPIDOS:")
    print("   sistema.estado_sistema() - Estado actual")
    print("   sistema.menu_calibracion() - Recalibrar")
    print("   sistema.detener_sistema() - Apagar")

if __name__ == "__main__":
    # Ejecutar sistema principal
    main()
    
    # Mostrar comandos si estamos en modo interactivo
    try:
        import __main__ as main_module
        if hasattr(main_module, '__file__'):
            # Script ejecutado desde archivo
            pass
        else:
            # Modo interactivo
            print("\n🔧 MODO INTERACTIVO - Puedes crear un sistema:")
            print("   sistema = SistemaIntegrado()")
            print("   sistema.iniciar_sistema()")
    except:
        pass