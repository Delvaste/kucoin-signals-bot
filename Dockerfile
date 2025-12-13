# Imagen base ligera
FROM python:3.10-slim

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements y instalarlos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todos los archivos del bot
COPY . .

# Exponer puerto para health check (opcional)
EXPOSE 8080

# Comando de inicio del bot
CMD ["python3", "futuros_bot.py"]
