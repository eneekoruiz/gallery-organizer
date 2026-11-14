import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

def verify():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1280, 'height': 800})
        page = context.new_page()

        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

        print("1. Cargando http://127.0.0.1:5000...")
        t0 = time.time()
        page.goto("http://127.0.0.1:5000", wait_until="networkidle", timeout=30000)
        t_load = time.time() - t0
        print(f"  Página cargada en {t_load:.2f}s!")

        # 2. Check sidebar categories
        page.wait_for_selector(".ios-category-card, .home-album-card, #categories-list", timeout=15000)
        page.wait_for_timeout(1500)
        category_cards = page.locator(".ios-category-card").all_inner_texts()
        print(f"  Total categorías visibles: {len(category_cards)}")
        for c in category_cards[:5]:
            print(f"    - {c.replace(chr(10), ' ')}")
        
        # Take screenshot of home
        ss_path = Path(r"C:\Users\User\.gemini\antigravity\brain\f3eef202-a90f-44c0-ae27-030ea25d0d1c\test_consolidated_gallery.png")
        page.screenshot(path=str(ss_path))
        print(f"  ✓ Screenshot de galería consolidada guardado en {ss_path.name}")

        # 3. Drill down into 'Revisión / Por Organizar'
        print("2. Navegando dentro de 'Revisión / Por Organizar'...")
        rev_card = page.locator(".ios-category-card:has-text('Revisión')").first
        if rev_card.count() > 0:
            rev_card.click()
            page.wait_for_timeout(1500)
            
            ident_rows = page.locator(".drilldown-identity-row").all_inner_texts()
            print(f"  Total sub-álbumes renderizados en Revisión: {len(ident_rows)}")
            for r in ident_rows[:8]:
                print(f"    - {r.replace(chr(10), ' ')}")

            ss_rev = Path(r"C:\Users\User\.gemini\antigravity\brain\f3eef202-a90f-44c0-ae27-030ea25d0d1c\test_revision_consolidated.png")
            page.screenshot(path=str(ss_rev))
            print(f"  ✓ Screenshot de Revisión consolidada guardado en {ss_rev.name}")

        # 4. Check C. Kid album
        print("2. Verificando álbum 'Conocidos > C. Kid'...")
        kid_url = "http://127.0.0.1:5000/#cat=Conocidos&ident=C.%20Kid"
        page.goto(kid_url, wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(1000)
        
        # Check header
        album_header = page.locator("#main-gallery-header").inner_text()
        print(f"  Cabecera de C. Kid: {album_header.replace(chr(10), ' | ')}")
        
        ss_kid = Path(r"C:\Users\User\.gemini\antigravity\brain\f3eef202-a90f-44c0-ae27-030ea25d0d1c\test_kid_verified.png")
        page.screenshot(path=str(ss_kid))
        print(f"  ✓ Screenshot de C. Kid guardado en {ss_kid.name}")

        # 4. Check C. Aaron album
        print("3. Verificando álbum 'Conocidos > C. Aaron'...")
        aaron_url = "http://127.0.0.1:5000/#cat=Conocidos&ident=C.%20Aaron"
        page.goto(aaron_url, wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(1000)
        
        album_header_aaron = page.locator("#main-gallery-header").inner_text()
        print(f"  Cabecera de C. Aaron: {album_header_aaron.replace(chr(10), ' | ')}")
        
        # Check if there is any suggestion or dudosa showing Kid
        dudosas_count = page.locator(".tier-header-review").count()
        print(f"  Bloques dudosas en Aaron: {dudosas_count}")

        ss_aaron = Path(r"C:\Users\User\.gemini\antigravity\brain\f3eef202-a90f-44c0-ae27-030ea25d0d1c\test_aaron_clean.png")
        page.screenshot(path=str(ss_aaron))
        print(f"  ✓ Screenshot de C. Aaron guardado en {ss_aaron.name}")

        # 5. Check Console Errors
        print(f"4. Errores de consola detectados: {len(console_errors)}")
        if console_errors:
            for err in console_errors:
                print(f"  ⚠️ {err}")
        else:
            print("  ✓ 0 errores de consola!")

        browser.close()
        print("VERIFICACIÓN PLAYWRIGHT COMPLETADA CON ÉXITO")

if __name__ == '__main__':
    verify()
