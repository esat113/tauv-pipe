#!/bin/bash
set -e

if [ -d "/sys/class/net/$DDS_INTERFACE" ]; then
    echo "[entrypoint] DDS interface: $DDS_INTERFACE (found)"
    envsubst < /app/dds_config.template.xml > /app/dds_config.xml
else
    echo "[entrypoint] WARNING: $DDS_INTERFACE not found, using auto-detect"
    cat > /app/dds_config.xml <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS>
  <Domain id="any">
    <General>
      <AllowMulticast>default</AllowMulticast>
      <MaxMessageSize>131000B</MaxMessageSize>
      <FragmentSize>32kB</FragmentSize>
    </General>
    <Discovery>
      <Peers>
        <Peer Address="${JETSON_IP}"/>
        <Peer Address="${PC_IP}"/>
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
EOF
fi

export CYCLONEDDS_URI=file:///app/dds_config.xml

exec python3 /app/main.py "$@"
