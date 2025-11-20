# Sistema de Procesamiento de Audio - DOA + Beamforming

## 📖 Descripción
Sistema integrado para procesamiento de audio en tiempo real que incluye:
- **PDG**: Visualización de señales de micrófonos
- **DOA**: Estimación de dirección de llegada (SRP-PHAT)
- **Beamforming**: Filtrado espacial direccional

## 🚀 Instalación

1. **Clonar o descargar** los archivos del proyecto
2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
3. Conectar el arreglo de micrófonos Respeaker 4-Mic array v2.0
4. Ejecutar el sistema:
   python SistemaIntegrado.py

## 🎛️ Configuración
Variables Importantes

## Audio (GestionDispositivos.py)
   sample_rate = 16000    # Frecuencia de muestreo
   blocksize = 1024       # Tamaño de bloque de audio
   channels = 6           # Canales de entrada

## DOA (DOA.py)
   radio = 0.0325         # Radio del array en metros
   sound_speed = 343.0    # Velocidad del sonido
   resolucion_grados = 1  # Resolución angular

## Gestión del arreglo (PDG.py)
   window_duration = 3.0  # Duración de ventana visualizada
   amplification_factor = 15.0  # Factor de amplificación

## Beamforming (Beamforming.py)
   buffer_duration = 5    # Segundos en buffer de espectrogramas

## 📊 Uso del Sistema
Inicio
- Ejecutar SistemaIntegrado.py
- El sistema detecta automáticamente el Respeaker
- Se abren 3 ventanas de visualización

Visualizaciones
 PDG: 4 gráficos de señales de micrófonos
 DOA: Gráfico polar con dirección estimada
 Beamforming: 2 espectrogramas comparativos

Guardado de Audio
 PDG: Botón "Guardar Audio" - guarda señales crudas
 Beamforming: Botón "Guardar Audio" - guarda señal beamformed 

## 🎤 Configuración de Hardware
 Arreglo de micrófonos
   Posiciones (coordenadas x,y en metros):
   Mic 1: [-0.0325, 0]    (Canal 2)
   Mic 2: [0, -0.0325]    (Canal 3)  
   Mic 3: [0.0325, 0]     (Canal 4)
   Mic 4: [0, 0.0325]     (Canal 5)
 Canales de Audio
   Canal 1: No usado
   Canales 2-5: Micrófonos 1-4
   Canal 6: No usado

## ⚙️ Solución de Problemas
   No detecta el Respeaker
      Verificar conexión USB
      Ejecutar como administrador si es necesario
      Verificar permisos de audio
   Error de dependencias
      Actualizar pip: python -m pip install --upgrade pip
      Reinstalar paquetes: pip install --force-reinstall -r requirements.txt

   Rendimiento pobre
      Reducir blocksize a 512
      Cerrar otras aplicaciones de audio
      Verificar uso de CPU

## 📁 Estructura de Archivos
   /
   ├── SistemaIntegrado.py    # Punto de entrada principal
   ├── GestionDispositivos.py # Gestión de audio
   ├── PDG.py                # Visualización de señales
   ├── DOA.py                # Dirección de llegada
   ├── Beamforming.py        # Beamforming y espectrogramas
   ├── requirements.txt      # Dependencias
   └── Audios_Guardados/     # Carpeta de salida de audio
      ├── Audios_Crudos/    # Señales originales
      └── Audios_Beamformed/ # Señales procesadas
