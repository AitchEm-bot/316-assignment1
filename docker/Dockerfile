FROM jupyter/pyspark-notebook:latest

USER root

# Install additional system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

# Install Python packages
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install Streamlit for web interface
RUN pip install --no-cache-dir streamlit

WORKDIR /home/jovyan/work
