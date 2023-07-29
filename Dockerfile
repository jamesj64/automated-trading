FROM python:3.10-slim

RUN apt-get update

RUN apt-get install -y --no-install-recommends git

RUN apt-get purge -y --auto-remove

RUN rm -rf /var/lib/apt/lists/

WORKDIR /app

ADD requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir src

COPY ./src ./src

ADD oanda.cfg .

CMD ["python", "-B", "./src/main.py"]
