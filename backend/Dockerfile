FROM python:3.11-slim

WORKDIR /app

# Instala dependências
COPY backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copia o código
COPY backend .

# Expondo a porta
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]