"""
CalibracionDOA.py - Herramienta simple de calibración
"""

import numpy as np
import time

class CalibradorDOA:
    def __init__(self, doa_system):
        self.doa = doa_system
        
    def calibrar_rapido(self, angulo_real=0, duracion=10):
        """
        Calibración rápida y simple
        """
        print(f"\n🎯 CALIBRACIÓN RÁPIDA")
        print(f"   • Fuente en {angulo_real}°")
        print(f"   • Habla/Música por {duracion} segundos")
        print(f"   • Distancia: 1-2 metros")
        print("   • Presiona Ctrl+C para cancelar\n")
        
        angulos_recolectados = []
        confianzas_recolectadas = []
        
        print("🔄 Recolectando datos...", end='', flush=True)
        
        start_time = time.time()
        try:
            while time.time() - start_time < duracion:
                angulo, confianza = self.doa.get_angulo_actual()
                if confianza > 0.4:  # Solo datos confiables
                    angulos_recolectados.append(angulo)
                    confianzas_recolectadas.append(confianza)
                print(".", end='', flush=True)
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n⚠️ Calibración cancelada")
            return None
        
        print("✅")
        
        if not angulos_recolectados:
            print("❌ No se capturaron datos válidos")
            print("   Verifica:")
            print("   - Volumen de la fuente")
            print("   - Distancia (1-2 metros)")
            print("   - Micrófonos conectados")
            return None
        
        # Calcular error sistemático
        angulos = np.array(angulos_recolectados)
        errores = []
        for angulo in angulos:
            error = (angulo - angulo_real + 180) % 360 - 180
            errores.append(error)
        
        error_mediano = np.median(errores)
        desviacion = np.std(errores)
        
        print(f"📊 Análisis:")
        print(f"   • Muestras: {len(angulos)}")
        print(f"   • Error sistemático: {error_mediano:.1f}°")
        print(f"   • Desviación: {desviacion:.1f}°")
        
        if abs(error_mediano) > 2:
            offset = -error_mediano
            self.doa.offset_calibracion = offset
            print(f"✅ Calibración aplicada:")
            print(f"   • Offset: {offset:.1f}°")
            print(f"   • Precisión esperada: ±{desviacion:.1f}°")
            return offset
        else:
            print("ℹ️ Sistema ya preciso")
            print(f"   • Error pequeño: {error_mediano:.1f}°")
            print(f"   • No se requiere calibración")
            self.doa.offset_calibracion = 0
            return 0