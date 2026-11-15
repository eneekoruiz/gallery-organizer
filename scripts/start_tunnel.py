# -*- coding: utf-8 -*-
"""
Inicia el túnel Cloudflare como proceso desacoplado persistente en Windows
utilizando el argumento nativo --logfile de cloudflared, captura la URL activa
y valida la respuesta HTTP 200.
"""
import os
import sys
import time
import subprocess
import re
import urllib.request

CLOUDFLARED_EXE = r"C:\Users\User\Documents\antigravity\keen-bell\cloudflared.exe"
LOG_FILE = os.path.abspath(os.path.join("scratch", "cloudflared.log"))
URL_FILE = os.path.abspath(os.path.join("scratch", "active_tunnel_url.txt"))

os.makedirs("scratch", exist_ok=True)

# 1. Matar instancias previas
subprocess.run(["taskkill", "/F", "/IM", "cloudflared.exe"], capture_output=True)
time.sleep(1)

# Eliminar log previo
if os.path.exists(LOG_FILE):
    try:
        os.remove(LOG_FILE)
    except Exception:
        pass

# 2. Iniciar cloudflared con --logfile nativo y DETACHED_PROCESS sin redirigir stdout/stderr a pipes de python
# Flags para desacoplar completamente el proceso en Windows:
# DETACHED_PROCESS (0x00000008) | CREATE_NEW_PROCESS_GROUP (0x00000200)
creationflags = 0x00000008 | 0x00000200

cmd = [
    CLOUDFLARED_EXE,
    "tunnel",
    "--url", "http://127.0.0.1:5000",
    "--logfile", LOG_FILE
]

proc = subprocess.Popen(
    cmd,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    creationflags=creationflags,
    close_fds=True
)

print(f"[TUNNEL] Proceso cloudflared lanzado con PID {proc.pid}")

# 3. Esperar a que cloudflared escriba la URL en el log
found_url = None
t0 = time.time()
while time.time() - t0 < 30:
    time.sleep(1.5)
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                matches = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
                if matches:
                    found_url = matches[0]
                    break
        except Exception:
            pass

if not found_url:
    print("[ERROR] No se pudo encontrar la URL en scratch/cloudflared.log")
    sys.exit(1)

print(f"[TUNNEL DETECTADO] URL: {found_url}")
with open(URL_FILE, "w", encoding="utf-8") as f:
    f.write(found_url)

# 4. Validar conectividad HTTP real
print("[VERIFICANDO] Comprobando acceso externo vía Cloudflare Edge...")
connected = False
for attempt in range(8):
    try:
        req = urllib.request.Request(
            found_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=8) as response:
            if response.status == 200:
                html = response.read().decode("utf-8", errors="ignore")
                if "Smart Gallery" in html:
                    print(f"[VERIFICADO OK] HTTP 200 recibido desde {found_url}")
                    connected = True
                    break
    except Exception as e:
        print(f" Intento {attempt+1}/8: esperando propagación DNS/Edge ({e})...")
        time.sleep(2.5)

if not connected:
    print("[WARN] No se pudo verificar HTTP 200 de inmediato, pero el proceso sigue activo.")
else:
    print(f"\n==========================================")
    print(f"URL ACTIVA Y OPERATIVA: {found_url}")
    print(f"==========================================\n")
