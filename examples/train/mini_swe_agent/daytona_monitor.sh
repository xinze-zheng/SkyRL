#!/bin/bash
# Daytona server resource monitor
# Runs on the Daytona server, captures CPU/mem/disk every 5 seconds
# Usage: bash daytona_monitor.sh /path/to/output.csv

OUTPUT=${1:-/tmp/daytona_metrics.csv}
INTERVAL=5

echo "timestamp,cpu_pct,mem_used_gb,mem_total_gb,mem_pct,disk_used_gb,disk_avail_gb,docker_containers,sandbox_count" > "$OUTPUT"

while true; do
    TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    
    # CPU usage (1-second sample)
    CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}')
    
    # Memory
    MEM_USED=$(free -g | awk '/^Mem:/{print $3}')
    MEM_TOTAL=$(free -g | awk '/^Mem:/{print $2}')
    MEM_PCT=$(free | awk '/^Mem:/{printf "%.1f", $3/$2*100}')
    
    # Disk (root)
    DISK_USED=$(df -BG / | awk 'NR==2{gsub("G",""); print $3}')
    DISK_AVAIL=$(df -BG / | awk 'NR==2{gsub("G",""); print $4}')
    
    # Docker container count
    CONTAINERS=$(docker ps -q 2>/dev/null | wc -l)
    
    # Daytona sandbox count via API
    SANDBOXES=$(curl -s -m 2 http://localhost:3000/api/sandbox -H "Authorization: Bearer ${DAYTONA_API_KEY}" 2>/dev/null | python3 -c "import json,sys; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)
    
    echo "${TS},${CPU},${MEM_USED},${MEM_TOTAL},${MEM_PCT},${DISK_USED},${DISK_AVAIL},${CONTAINERS},${SANDBOXES}" >> "$OUTPUT"
    
    sleep $INTERVAL
done
