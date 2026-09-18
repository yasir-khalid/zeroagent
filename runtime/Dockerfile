FROM python:3.12-slim

# The image is deliberately small but has the two programs exposed to the agent.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends bash git \
    && useradd --create-home --shell /bin/bash sandbox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
USER sandbox
