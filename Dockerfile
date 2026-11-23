FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/faults.json data/machines.json data/
COPY app.py admin.py demo_chat.py ./
CMD ["python", "app.py"]
