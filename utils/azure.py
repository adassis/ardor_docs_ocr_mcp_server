# =============================================================
# utils/azure.py — Fonctions d'appel à l'API Azure OCR
# =============================================================
# Contient TOUTE la logique de communication avec Azure.
# Importé par tous les outils qui ont besoin d'OCR (url, pipedrive, etc.)
# Aucune dépendance vers d'autres modules internes du projet.
# =============================================================

import json
import re
import time
import requests


def _sleep_retry_after(resp, default_seconds=5):
    """
    Gère les pauses lors d'une erreur 429 (trop de requêtes) renvoyée par Azure.

    Azure peut indiquer dans sa réponse combien de secondes attendre,
    soit via l'en-tête HTTP 'Retry-After', soit dans le message d'erreur JSON.
    Si aucune indication, on attend default_seconds.

    Args:
        resp           : objet réponse requests
        default_seconds: secondes à attendre si Azure ne précise pas
    """
    # Tentative 1 : lire l'en-tête HTTP standard Retry-After
    ra = resp.headers.get("Retry-After")
    if ra:
        try:
            time.sleep(int(ra))
            return
        except Exception:
            pass

    # Tentative 2 : parser le message d'erreur JSON d'Azure
    # Format observé : "retry after X seconds"
    try:
        msg = resp.json().get("error", {}).get("message", "") or ""
        m = re.search(r"retry after\s+(\d+)\s+seconds", msg, re.IGNORECASE)
        if m:
            time.sleep(int(m.group(1)))
            return
    except Exception:
        pass

    # Fallback : attente par défaut
    time.sleep(default_seconds)


def azure_read_analyze(file_bytes: bytes, endpoint: str, key: str, language: str = "auto") -> dict:
    """
    Envoie un fichier binaire à Azure Computer Vision OCR et retourne le résultat.

    Fonctionnement en 2 phases :
    1. POST du fichier → Azure répond 202 + une URL de suivi (Operation-Location)
    2. Polling de cette URL jusqu'à obtenir le statut "succeeded"

    Gère automatiquement les erreurs 429 avec retry exponentiel.

    Args:
        file_bytes : contenu binaire du fichier (image ou PDF)
        endpoint   : URL Azure Computer Vision (sans / final)
        key        : clé API Azure
        language   : code langue ISO ('fr', 'en'...) ou 'auto'

    Returns:
        dict : réponse JSON complète d'Azure avec le texte reconnu

    Raises:
        RuntimeError  : erreur HTTP non récupérable
        TimeoutError  : polling dépassé (60 tentatives)
    """
    analyze_url = endpoint + "/vision/v3.2/read/analyze"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/octet-stream",
        # On envoie le fichier en binaire brut (pas de base64, pas de multipart)
    }
    params = {}
    if language and language.lower() != "auto":
        params["language"] = language
        # Si "auto" : on n'envoie pas le paramètre, Azure détecte la langue seul

    # ── Phase 1 : POST du fichier avec retry sur 429 ──────────
    for attempt in range(6):
        r = requests.post(
            analyze_url, headers=headers, params=params,
            data=file_bytes, timeout=120
        )
        if r.status_code in (200, 202):
            break  # Succès : 202 = traitement asynchrone lancé
        if r.status_code == 429:
            # Limite de taux Azure dépassée → on attend et on réessaie
            _sleep_retry_after(r, default_seconds=min(5 * (attempt + 1), 30))
            continue
        raise RuntimeError(f"Azure analyze error {r.status_code}: {r.text[:400]}")
    else:
        raise RuntimeError(f"Azure analyze error 429 après 6 tentatives: {r.text[:400]}")

    # ── Récupération de l'URL de polling ─────────────────────
    op_location = r.headers.get("Operation-Location")
    if not op_location:
        # Cas rare : Azure répond directement (sans async)
        try:
            return r.json()
        except Exception:
            raise RuntimeError("Azure: Operation-Location absent et pas de JSON.")

    # ── Phase 2 : Polling jusqu'à "succeeded" ─────────────────
    poll_headers = {"Ocp-Apim-Subscription-Key": key}
    wait = 2  # Délai initial entre deux polls (en secondes)

    for _ in range(60):  # 60 tentatives max (~10 minutes)
        rr = requests.get(op_location, headers=poll_headers, timeout=60)

        if rr.status_code == 429:
            _sleep_retry_after(rr, default_seconds=wait)
            wait = min(int(wait * 1.5), 10)
            continue

        if rr.status_code != 200:
            time.sleep(wait)
            wait = min(int(wait * 1.5), 10)
            continue

        data = rr.json()
        st = (data.get("status") or "").lower()

        if st == "succeeded":
            return data  # ✅ OCR terminé avec succès

        if st == "failed":
            raise RuntimeError(f"Azure OCR a échoué: {json.dumps(data)[:500]}")

        # Statut "running" ou "notStarted" → on attend et on repoll
        time.sleep(wait)
        wait = min(int(wait * 1.5), 10)

    raise TimeoutError("Azure OCR: délai de polling dépassé (60 tentatives)")


def extract_text_from_azure_result(result_json: dict):
    """
    Extrait le texte et la structure page par page depuis la réponse JSON d'Azure.

    Gère 2 formats de réponse Azure :
    - 'readResults' : format API v3.2 (actuel)
    - 'pages'       : format API v4+ (futur / Document Intelligence)

    Args:
        result_json : dict JSON complet retourné par azure_read_analyze()

    Returns:
        tuple (text_full, pages_struct) :
        - text_full    : str, tout le texte concaténé ligne par ligne
        - pages_struct : list de dicts {"page_index": int, "lines": [str, ...]}
    """
    text_lines = []
    pages_struct_local = []
    ar = result_json.get("analyzeResult", {})

    # ── Format v3.2 : readResults ─────────────────────────────
    read_results = ar.get("readResults") or []
    if read_results:
        for i, page in enumerate(read_results):
            page_lines = []
            for line in page.get("lines", []):
                t = line.get("text", "")
                if t:
                    text_lines.append(t)
                    page_lines.append(t)
            pages_struct_local.append({"page_index": i + 1, "lines": page_lines})
        return "\n".join(text_lines).strip(), pages_struct_local

    # ── Format v4+ : pages ────────────────────────────────────
    pages = ar.get("pages") or []
    if pages:
        for i, page in enumerate(pages):
            page_lines = []
            for line in page.get("lines", []):
                txt = line.get("content") or line.get("text") or ""
                if not txt and "words" in line:
                    txt = " ".join([
                        w.get("content", "")
                        for w in line.get("words", [])
                        if w.get("content")
                    ])
                if txt:
                    text_lines.append(txt)
                    page_lines.append(txt)
            pages_struct_local.append({"page_index": i + 1, "lines": page_lines})
        return "\n".join(text_lines).strip(), pages_struct_local

    # Aucun texte trouvé
    return "", []