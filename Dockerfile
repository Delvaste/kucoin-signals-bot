# Imagen base de Python
FROM python:3.10-slim

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements e instalarlos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código al contenedor
COPY . .

# Comando de inicio: nuestro bot
CMD ["python", "futuros_liquidez.py"]
