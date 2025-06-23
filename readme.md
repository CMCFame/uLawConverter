# μ-law to WAV Converter & Transcriber - Cloud Edition

Una aplicación web para convertir archivos de audio μ-law a formato WAV y transcribirlos usando IA.

## 🚀 Despliegue en Streamlit Cloud

### 1. Preparar el Repositorio
```bash
# Estructura de archivos necesaria:
├── app.py                    # Aplicación principal
├── requirements.txt          # Dependencias de Python
├── .streamlit/
│   └── config.toml          # Configuración de Streamlit
└── README.md                # Este archivo
```

### 2. Configurar Secrets
En Streamlit Cloud, ve a **Settings > Secrets** y agrega:

```toml
DEEPGRAM_API_KEY = "tu_api_key_de_deepgram_aqui"
```

### 3. Deploy
1. Haz push del código a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repositorio de GitHub
4. La app se desplegará automáticamente

## 📋 Funcionalidades

- **Upload de archivos**: Individual o por lotes
- **Conversión μ-law → WAV**: Procesamiento en tiempo real
- **Transcripción IA**: Usando Deepgram Nova-2
- **Monitoreo en vivo**: Logs y progreso detallado
- **Descarga de resultados**: CSV con transcripciones y ZIP con WAVs
- **Procesamiento seguro**: Todo en memoria, sin almacenamiento en servidor

## 🔧 Configuración Local (Opcional)

```bash
# Instalar dependencias
pip install -r requirements.txt

# Configurar API key
echo 'DEEPGRAM_API_KEY = "tu_api_key"' > .streamlit/secrets.toml

# Ejecutar localmente
streamlit run app.py
```

## 📊 Límites de Streamlit Cloud

- **Tamaño de archivo**: Máximo 200MB por archivo
- **Memoria**: Limitada por los recursos de cloud
- **Sesiones**: Temporales, se reinician periódicamente
- **Concurrencia**: Limitada por el plan de Streamlit Cloud

## 🎵 Formatos Soportados

- `.ulaw` - μ-law estándar
- `.ul` - μ-law alternativo  
- `.au` - Audio Unix
- `.raw` - Datos raw μ-law

## 🔒 Seguridad

- Los archivos se procesan únicamente en memoria
- No se almacena información en el servidor
- Las transcripciones están disponibles solo durante la sesión
- API key protegida mediante Streamlit Secrets

## 📞 Soporte

Para problemas o preguntas:
1. Revisa que el API key de Deepgram esté configurado correctamente
2. Verifica que los archivos sean formato μ-law válido
3. Comprueba los logs en tiempo real para detalles de errores