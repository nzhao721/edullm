#!/usr/bin/env bash
# Report free GPUs per oat node (run on a FarmShare login node).
set -Eeuo pipefail

source /etc/profile.d/z00_lmod.sh 2>/dev/null || true

echo "=== Free GPUs per oat node ==="
printf "%-8s %-10s %5s %5s %5s %9s %10s\n" NODE STATE TOTAL USED FREE CPU_FREE MEM_FREE_GB
for n in oat-01 oat-02 oat-03 oat-04 oat-05 oat-06; do
  info=$(scontrol show node "$n" 2>/dev/null)
  state=$(echo "$info" | awk '{for(i=1;i<=NF;i++) if($i~/^State=/) print substr($i,7)}')
  gpu_total=$(echo "$info" | awk '{for(i=1;i<=NF;i++) if($i~/^Gres=/) print $i}' | sed -n 's/.*gpu:\([0-9]*\).*/\1/p')
  cpualloc=$(echo "$info" | awk '{for(i=1;i<=NF;i++) if($i~/^CPUAlloc=/) print substr($i,10)}')
  cputot=$(echo "$info" | awk '{for(i=1;i<=NF;i++) if($i~/^CPUTot=/) print substr($i,8)}')
  allocmem=$(echo "$info" | awk '{for(i=1;i<=NF;i++) if($i~/^AllocMem=/) print substr($i,10)}')
  realmem=$(echo "$info" | awk '{for(i=1;i<=NF;i++) if($i~/^RealMemory=/) print substr($i,13)}')
  gpu_used=0
  while read -r tres; do
    g=$(echo "$tres" | sed -n 's/.*gpu:\([0-9]*\).*/\1/p')
    gpu_used=$((gpu_used + ${g:-0}))
  done < <(squeue -p gpu -t RUNNING -w "$n" -h -o "%b" 2>/dev/null)
  gpu_free=$((gpu_total - gpu_used))
  cpu_free=$((cputot - cpualloc))
  mem_free_gb=$(( (realmem - allocmem) / 1024 ))
  printf "%-8s %-10s %5d %5d %5d %9d %10d\n" "$n" "$state" "$gpu_total" "$gpu_used" "$gpu_free" "$cpu_free" "$mem_free_gb"
done
