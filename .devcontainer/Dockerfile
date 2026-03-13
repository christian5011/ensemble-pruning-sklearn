FROM mcr.microsoft.com/devcontainers/python:3.12

# Install system utilities
RUN apt-get update && \
	apt-get install -y \
	vim && \
	rm -rf /var/lib/apt/lists/*

# Install demo and development extras.
COPY requirements-dev.txt /tmp/
RUN pip install --upgrade pip && \
	pip install -r /tmp/requirements-dev.txt

USER vscode
