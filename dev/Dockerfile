FROM mcr.microsoft.com/devcontainers/python:3.12

# Install system utilities
RUN apt-get update && \
	apt-get install -y \
	vim && \
	rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /tmp/
RUN pip install --upgrade pip && \
	pip install -r /tmp/requirements.txt

USER vscode