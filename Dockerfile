FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUTF8=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "futuros_liquidez.py"]
