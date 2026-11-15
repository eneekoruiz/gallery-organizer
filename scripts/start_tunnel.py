# -*- coding: utf-8 -*-
"""
Inicia el túnel Cloudflare en segundo plano como proceso desacoplado persistente,
captura la URL activa y la guarda en scratch/active_tunnel_url.txt.
"""
import os
import sys
import time
import subprocess
import re

CLOUDFLARED_EXE = r"C:\Users\User\Documents\antigravity\keen-bell\cloudflared.exe"
OUTPUT_FILE = os.path.join("scratch", "cloudflared.log")
URL_FILE = os.path.join("scratch", "active_tunnel_url.txt")

os.makedirs("scratch", exist_ok=True)

# 1. Matar instancias previas de cloudflared
subprocess.run(["taskkill", "/F", "/IM", "cloudflared.exe"], capture_output=True)
time.sleep(1)

# 2. Iniciar cloudflared redirigiendo a archivo de log con protocolo http2 estable
with open(OUTPUT_FILE, "w", encoding="utf-8") as log_fp:
    proc = subprocess.Popen(
        [CLOUDFLARED_EXE, "tunnel", "--protocol", "http2", "--url", "http://127.0.0.1:5000"],
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008  # DETACHED_PROCESS
    )

print(f"[TUNNEL] cloudflared iniciado con PID {proc.pid}")

# 3. Monitorear el archivo de log hasta encontrar la URL
found_url = None
t0 = time.time()
while time.time() - t0 < 25:
    time.sleep(1)
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                matches = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
                if matches:
                    found_url = matches[0]
                    break
        except Exception:
            pass

if found_url:
    with open(URL_FILE, "w", encoding="utf-8") as f:
        f.write(found_url)
    print(f"[TUNNEL ACTIVO] URL: {found_url}")
else:
    print("[ERROR] No se pudo obtener la URL en el tiempo límite.")
