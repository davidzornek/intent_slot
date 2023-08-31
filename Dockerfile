FROM python:3.8

WORKDIR /work

COPY requirements.txt .
COPY .gitignore .
COPY config.yaml .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY intent_slot/ .

EXPOSE 5000

CMD ["python", "api.py"]
