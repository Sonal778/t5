FROM python:3.8

RUN apt-get update

COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY . .

RUN mkdir ./Server/Model
WORKDIR ./Server/Model
RUN pip3 install gdown && \
    gdown https://drive.google.com/uc?id=1-1sx4qC08MgE7XZwd0MwFQpvYtMrXxGZ && \
    gdown https://drive.google.com/uc?id=1Lk3dflSpm_c0vUs88tY4GjETYqpI8Gfb

WORKDIR /

RUN mkdir ./Server/Token
WORKDIR ./Server/Token
RUN pip3 install gdown && \
    gdown https://drive.google.com/uc?id=1-3p_g-iuMG1brqZxEHPL2eb6Sk1vaDpS && \
    gdown https://drive.google.com/uc?id=1-16h1St1lyvvZKyot-jggeDtXVn3RarT && \
    gdown https://drive.google.com/uc?id=1-1XAA6TSFQO5a1SWattirsoIsMtBG4oO

WORKDIR /

EXPOSE 8000

EXPOSE $PORT

WORKDIR ./Server

CMD ["gunicorn"  , "-b", "0.0.0.0:8000", "server:app"]
