FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gettext-base \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY dds/ /app/dds/
COPY main.py controller.py mask_processor.py dds_interface.py debug_stream.py /app/
COPY config.yaml /app/config.yaml
COPY dds_config.template.xml /app/dds_config.template.xml
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
