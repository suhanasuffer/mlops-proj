FROM python:3.10-slim-bookworm

WORKDIR /app

#install dependencies
COPY requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install dvc[s3] awscli

#copy source code
COPY ..

#run full dvc pipeline
RUN dvc repro

EXPOSE 8080

#run main app
CMD ["python", "script/analyze_damage.py"]
