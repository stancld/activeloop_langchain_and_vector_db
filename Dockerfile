FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        gcc \
        python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && . $HOME/.cargo/env

COPY requirements.txt ./
COPY constraints.txt ./
RUN /root/.cargo/bin/uv venv /opt/venv \
    && . /opt/venv/bin/activate \
    && /root/.cargo/bin/uv pip install --no-cache -r requirements.txt -c constraints.txt

COPY notebooks notebooks

ENV PATH="/opt/venv/bin:$PATH"
