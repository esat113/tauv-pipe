#!/bin/bash
set -e

: "${DDS_INTERFACE:=lo}"
: "${JETSON_IP:=localhost}"
: "${PC_IP:=localhost}"
: "${DDS_PEER_1:=${JETSON_IP}}"
: "${DDS_PEER_2:=${PC_IP}}"
export DDS_INTERFACE JETSON_IP PC_IP DDS_PEER_1 DDS_PEER_2

if [ -d "/sys/class/net/$DDS_INTERFACE" ]; then
    echo "[entrypoint] DDS interface: $DDS_INTERFACE (found)"
    envsubst '${DDS_INTERFACE} ${DDS_PEER_1} ${DDS_PEER_2}' < /app/dds_config.template.xml > /app/dds_config.xml
else
    echo "[entrypoint] WARNING: $DDS_INTERFACE not found, using auto-detect"
    cat > /app/dds_config.xml <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS>
  <Domain id="any">
    <General>
      <AllowMulticast>default</AllowMulticast>
      <MaxMessageSize>8388608B</MaxMessageSize>
      <FragmentSize>32kB</FragmentSize>
    </General>
    <Discovery>
      <Peers>
        <Peer Address="localhost"/>
        <!-- Real vehicle peers:
        <Peer Address="192.168.1.100"/>
        <Peer Address="192.168.1.101"/>
        -->
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
EOF
fi

export CYCLONEDDS_URI=file:///app/dds_config.xml
echo "[entrypoint] DDS_PEER_1=${DDS_PEER_1}"
echo "[entrypoint] DDS_PEER_2=${DDS_PEER_2}"

exec python3 /app/main.py "$@"
