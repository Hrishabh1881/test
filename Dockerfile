# syntax=docker/dockerfile:1
FROM python:3.10

# Set environment variables to non-interactive and avoid writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /code

COPY requirements.txt /code/

RUN apt update -y \
    && apt upgrade -y \
    && pip install -r requirements.txt

COPY . /code/



