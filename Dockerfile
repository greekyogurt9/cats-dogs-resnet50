FROM tensorflow/tensorflow:2.20.0

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY cats_dogs_resnet50_compat.keras .

EXPOSE 7860

CMD ["python", "app.py"]