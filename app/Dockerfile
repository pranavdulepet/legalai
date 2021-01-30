FROM python:3.8-buster

RUN apt update && apt install -y vim wget curl

WORKDIR /home

COPY requirements.txt .

RUN ["python", "-m", "pip", "install", "--upgrade", "pip"]
RUN ["python", "-m", "pip", "install", "--no-cache-dir", "-r", "requirements.txt"]
RUN ["python", "-c", "import nltk; nltk.download('stopwords')"]
RUN ["python", "-m", "spacy", "download", "en_core_web_md"]

COPY . .

RUN chmod +x start.sh

EXPOSE 8000

# on startup
CMD ["./start.sh"]
