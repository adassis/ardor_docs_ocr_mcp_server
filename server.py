# =============================================================
# server.py — Serveur MCP Azure OCR
# Expose un outil "read_document_ocr" connectable à Dust
# =============================================================

# ── IMPORTS ──────────────────────────────────────────────────
import os           # Lecture des variables d'environnement (credentials Azure)
import json         # Sérialisation des résultats en JSON pour Dust
import re           # Expressions régulières (extraction clé-valeur, nettoyage)
import mimetypes    # Détection du type de fichier depuis le Content-Type HTTP
import requests     # Appels HTTP : téléchargement du fichier + API Azure
import time         # Pauses entre les tentatives (retry / polling)

from mcp.server.fastmcp import FastMCP
# ↑ FastMCP est le framework haut-niveau du SDK MCP Python.
# Il gère automatiquement le protocole MCP, la définition des outils,
# et le transport HTTP vers Dust.

# ── INITIALISATION DU SERVEUR MCP ────────────────────────────
mcp = FastMCP(
    name="azure-ocr-server",
    # ↑ Nom affiché dans Dust lors de l'ajout du serveur
    instructions=(
        "Serveur OCR Azure Computer Vision. "
        "Utilisez l'outil read_document_ocr pour extraire le texte "
        "d'un document image ou PDF depuis une URL publique."
    )
    # ↑ Description affichée dans l'interface Dust
)

# ── LECTURE DES CREDENTIALS AZURE ────────────────────────────
# Ces valeurs seront définies comme variables d'environnement sur Railway
# Elles ne sont JAMAIS exposées dans le code ni envoyées à Dust
AZURE_VISION_ENDPOINT = os.environ.get("AZURE_VISION_ENDPOINT", "").rstrip("/")
AZURE_VISION_KEY      = os.environ.get("AZURE_VISION_KEY", "")
MCP_BEARER_TOKEN      = os.environ.get("MCP_BEARER_TOKEN", "")
# ↑ Token de sécurité optionnel : si défini, le serveur vérifie que
# chaque requête entrante (de Dust) contient ce token dans le header Authorization


# =============================================================
# FONCTIONS UTILITAIRES — Reprises de votre code Zapier
# =============================================================

def download_file_from_url(url: str):
    """
    Télécharge un fichier depuis une URL publique.
    Vérifie que l'URL renvoie bien un fichier binaire (pas du HTML).
    Retourne : (contenu_bytes, content_type, extension)
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    # ↑ Simule un navigateur web pour éviter les blocages par les CDN
    
    r = requests.get(url, headers=headers, timeout=90, allow_redirects=True)
    # ↑ timeout=90 : on attend max 90 secondes (fichiers lourds sur connexion lente)
    # allow_redirects=True : suit les redirections (ex: URLs courtes, signed URLs S3)
    
    if r.status_code != 200:
        raise RuntimeError(
            f"Echec téléchargement URL (HTTP {r.status_code}): {r.text[:400]}"
        )

    content = r.content
    ct = r.headers.get("Content-Type", "") or ""

    # Sécurité : rejeter si on reçoit une page HTML au lieu d'un fichier
    if ct.startswith("text/html") or b"<html" in content[:512].lower():
        raise RuntimeError(
            f"L'URL a renvoyé du HTML. Content-Type={ct}. Extrait: {content[:200]!r}"
        )

    # Déterminer l'extension du fichier
    ext = mimetypes.guess_extension(ct.split(";")[0].strip()) or ""
    if not ext:
        # Fallback : chercher l'extension dans l'URL elle-même
        m = re.search(r"\.([a-zA-Z0-9]{2,5})(?:\?|$)", url)
        if m:
            ext = "." + m.group(1).lower()

    return content, ct, ext


def _sleep_retry_after(resp, default_seconds=5):
    """
    Gère le délai de retry quand Azure renvoie une erreur 429 (rate limit).
    Azure peut indiquer le délai à attendre dans le header "Retry-After"
    ou dans le corps JSON du message d'erreur.
    """
    ra = resp.headers.get("Retry-After")
    if ra:
        try:
            time.sleep(int(ra))
            return
        except Exception:
            pass
    try:
        msg = resp.json().get("error", {}).get("message", "") or ""
        m = re.search(r"retry after\s+(\d+)\s+seconds", msg, re.IGNORECASE)
        if m:
            time.sleep(int(m.group(1)))
            return
    except Exception:
        pass
    time.sleep(default_seconds)
    # ↑ Fallback : attend le nombre de secondes par défaut


def azure_read_analyze(file_bytes: bytes, endpoint: str, key: str, language: str = "auto") -> dict:
    """
    Envoie le fichier à l'API Azure Computer Vision Read v3.2.
    Gère le pattern asynchrone : POST pour soumettre → GET pour poller le résultat.
    
    L'API Azure OCR fonctionne en 2 temps :
    1. POST /vision/v3.2/read/analyze → Azure accepte et renvoie une URL de suivi
    2. GET {Operation-Location} → on interroge jusqu'à obtenir "succeeded" ou "failed"
    """
    analyze_url = endpoint + "/vision/v3.2/read/analyze"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        # ↑ Header d'authentification spécifique à Azure Cognitive Services
        "Content-Type": "application/octet-stream",
        # ↑ On envoie les bytes bruts du fichier (pas de base64, pas de JSON)
    }
    params = {}
    if language and language.lower() != "auto":
        params["language"] = language
        # ↑ Paramètre optionnel : si on connaît la langue, ça améliore la précision OCR

    # Boucle de retry pour les erreurs 429 (rate limit Azure)
    for attempt in range(6):
        r = requests.post(analyze_url, headers=headers, params=params, data=file_bytes, timeout=120)
        if r.status_code in (200, 202):
            break  # ← Succès : Azure a accepté la tâche
        if r.status_code == 429:
            _sleep_retry_after(r, default_seconds=min(5 * (attempt + 1), 30))
            continue  # ← Rate limit : on attend et on réessaie
        raise RuntimeError(f"Azure analyze error {r.status_code}: {r.text[:400]}")
    else:
        raise RuntimeError(f"Azure analyze error 429 après 6 tentatives: {r.text[:400]}")

    # Récupération de l'URL de polling asynchrone
    op_location = r.headers.get("Operation-Location")
    if not op_location:
        try:
            return r.json()  # ← Certaines requêtes retournent directement le résultat
        except Exception:
            raise RuntimeError("Azure: Operation-Location absent et pas de JSON.")

    # Polling : on interroge Azure jusqu'à ce que l'analyse soit terminée
    poll_headers = {"Ocp-Apim-Subscription-Key": key}
    wait = 2  # Délai initial entre chaque poll : 2 secondes
    for _ in range(60):  # Max 60 tentatives (~10 minutes max)
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
            return data  # ← Résultat final disponible
        if st == "failed":
            raise RuntimeError(f"Azure OCR a échoué: {json.dumps(data)[:500]}")
        
        # Statut "running" ou "notStarted" → on attend et on réessaie
        time.sleep(wait)
        wait = min(int(wait * 1.5), 10)
        # ↑ Backoff exponentiel : 2s → 3s → 4.5s → 6.75s → ... max 10s

    raise TimeoutError("Azure OCR: délai de polling dépassé (60 tentatives)")


def extract_text_from_azure_result(result_json: dict):
    """
    Extrait le texte et la structure par page depuis la réponse JSON d'Azure.
    Gère les 2 formats possibles de réponse Azure (readResults et pages).
    Retourne : (texte_complet_string, liste_de_pages_structurées)
    """
    text_lines = []
    pages_struct_local = []

    ar = result_json.get("analyzeResult", {})

    # FORMAT A : API Read v3.2 classique → structure "readResults"
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

    # FORMAT B : API Document Intelligence → structure "pages"
    pages = ar.get("pages") or []
    if pages:
        for i, page in enumerate(pages):
            page_lines = []
            for line in page.get("lines", []):
                txt = line.get("content") or line.get("text") or ""
                if not txt and "words" in line:
                    txt = " ".join([
                        w.get("content", "") for w in line.get("words", [])
                        if w.get("content")
                    ])
                if txt:
                    text_lines.append(txt)
                    page_lines.append(txt)
            pages_struct_local.append({"page_index": i + 1, "lines": page_lines})
        return "\n".join(text_lines).strip(), pages_struct_local

    return "", []  # ← Aucun texte trouvé


def normalize_key(k: str) -> str:
    """Normalise une clé extraite : supprime la ponctuation, met en minuscules."""
    k = k.strip().strip(":").strip("-").strip("=").strip()
    k = re.sub(r"\s+", " ", k)  # ← Remplace les espaces multiples par un seul
    return k.lower()


def normalize_val(v: str) -> str:
    """Normalise une valeur extraite : supprime les espaces superflus."""
    v = v.strip()
    v = re.sub(r"\s+", " ", v)
    return v


def safe_round(v, digits=4):
    """Arrondit un nombre flottant à n décimales de manière sécurisée."""
    if isinstance(v, (int, float)):
        return round(float(v), digits)
    return None


def percentile(sorted_values, p):
    """Calcule le p-ième percentile d'une liste triée."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = (len(sorted_values) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def confidence_distribution(confidences):
    """
    Calcule la distribution statistique des scores de confiance OCR.
    Retourne : moyenne, min, max, médiane, percentiles, stddev, et segmentation par seuils.
    """
    if not confidences:
        return {
            "count": 0, "avg": None, "min": None, "max": None,
            "median": None, "p10": None, "p25": None, "p75": None, "p90": None,
            "stddev": None,
            "low_lt_0_50": {"count": 0, "ratio": None},
            "low_lt_0_60": {"count": 0, "ratio": None},
            "medium_0_60_0_85": {"count": 0, "ratio": None},
            "high_gte_0_85": {"count": 0, "ratio": None},
            "very_high_gte_0_95": {"count": 0, "ratio": None}
        }

    vals = sorted([float(x) for x in confidences])
    count = len(vals)
    avg = sum(vals) / count
    variance = sum((x - avg) ** 2 for x in vals) / count
    stddev = variance ** 0.5

    # Segmentation par seuils de confiance
    c_lt_050 = sum(1 for x in vals if x < 0.50)   # Très faible
    c_lt_060 = sum(1 for x in vals if x < 0.60)   # Faible
    c_mid    = sum(1 for x in vals if 0.60 <= x < 0.85)  # Moyen
    c_hi     = sum(1 for x in vals if x >= 0.85)  # Élevé
    c_vhi    = sum(1 for x in vals if x >= 0.95)  # Très élevé

    return {
        "count": count,
        "avg": safe_round(avg),
        "min": safe_round(vals[0]),
        "max": safe_round(vals[-1]),
        "median": safe_round(percentile(vals, 0.50)),
        "p10": safe_round(percentile(vals, 0.10)),
        "p25": safe_round(percentile(vals, 0.25)),
        "p75": safe_round(percentile(vals, 0.75)),
        "p90": safe_round(percentile(vals, 0.90)),
        "stddev": safe_round(stddev),
        "low_lt_0_50":       {"count": c_lt_050, "ratio": safe_round(c_lt_050 / count)},
        "low_lt_0_60":       {"count": c_lt_060, "ratio": safe_round(c_lt_060 / count)},
        "medium_0_60_0_85":  {"count": c_mid,    "ratio": safe_round(c_mid    / count)},
        "high_gte_0_85":     {"count": c_hi,     "ratio": safe_round(c_hi     / count)},
        "very_high_gte_0_95":{"count": c_vhi,    "ratio": safe_round(c_vhi    / count)},
    }


def extract_ocr_metrics(result_json: dict):
    """
    Extrait les métriques OCR détaillées par page (nombre de mots,
    scores de confiance, dimensions de la page, angle de rotation...).
    """
    ar = result_json.get("analyzeResult", {})
    global_word_confidences = []
    pages_metrics = []
    global_word_count = 0
    global_line_count = 0
    global_page_count = 0

    read_results = ar.get("readResults") or []
    if read_results:
        global_page_count = len(read_results)
        for i, page in enumerate(read_results):
            page_word_confidences = []
            page_words = []
            page_lines_count = 0
            page_angle  = page.get("angle")
            page_width  = page.get("width")
            page_height = page.get("height")
            page_unit   = page.get("unit")

            for line in page.get("lines", []) or []:
                page_lines_count += 1
                for word in line.get("words", []) or []:
                    txt  = word.get("text") or word.get("content") or ""
                    conf = word.get("confidence")
                    if txt:
                        page_words.append(txt)
                    if isinstance(conf, (int, float)):
                        page_word_confidences.append(float(conf))
                        global_word_confidences.append(float(conf))

            global_line_count += page_lines_count
            global_word_count += len(page_words)
            page_text = " ".join(page_words).strip()
            dist = confidence_distribution(page_word_confidences)

            pages_metrics.append({
                "page_index": i + 1,
                "line_count": page_lines_count,
                "word_count": len(page_words),
                "char_count_estimated": len(page_text),
                "avg_word_length": safe_round(
                    sum(len(w) for w in page_words) / len(page_words)
                ) if page_words else None,
                "confidence": dist,
                "layout": {
                    "width": page_width, "height": page_height,
                    "unit": page_unit,   "angle": page_angle
                }
            })

        global_dist = confidence_distribution(global_word_confidences)
        return {
            "source_format": "readResults",
            "summary": {
                "page_count": global_page_count,
                "line_count": global_line_count,
                "word_count": global_word_count,
                "confidence": global_dist
            },
            "pages": pages_metrics
        }

    pages = ar.get("pages") or []
    if pages:
        global_page_count = len(pages)
        for i, page in enumerate(pages):
            page_word_confidences = []
            page_words = []
            page_lines_count = 0
            page_angle  = page.get("angle")
            page_width  = page.get("width")
            page_height = page.get("height")
            page_unit   = page.get("unit")

            for line in page.get("lines", []) or []:
                page_lines_count += 1
                line_words = line.get("words", []) or []
                if line_words:
                    for word in line_words:
                        txt  = word.get("content") or word.get("text") or ""
                        conf = word.get("confidence")
                        if txt:
                            page_words.append(txt)
                        if isinstance(conf, (int, float)):
                            page_word_confidences.append(float(conf))
                            global_word_confidences.append(float(conf))
                else:
                    txt = line.get("content") or line.get("text") or ""
                    if txt:
                        page_words.extend(re.findall(r"[^\s]+", txt))

            page_level_words = page.get("words", []) or []
            if page_level_words:
                page_words = []
                page_word_confidences = []
                for word in page_level_words:
                    txt  = word.get("content") or word.get("text") or ""
                    conf = word.get("confidence")
                    if txt:
                        page_words.append(txt)
                    if isinstance(conf, (int, float)):
                        page_word_confidences.append(float(conf))
                global_word_confidences.extend(page_word_confidences)

            global_line_count += page_lines_count
            global_word_count += len(page_words)
            page_text = " ".join(page_words).strip()
            dist = confidence_distribution(page_word_confidences)

            pages_metrics.append({
                "page_index": i + 1,
                "line_count": page_lines_count,
                "word_count": len(page_words),
                "char_count_estimated": len(page_text),
                "avg_word_length": safe_round(
                    sum(len(w) for w in page_words) / len(page_words)
                ) if page_words else None,
                "confidence": dist,
                "layout": {
                    "width": page_width, "height": page_height,
                    "unit": page_unit,   "angle": page_angle
                }
            })

        global_dist = confidence_distribution(global_word_confidences)
        return {
            "source_format": "pages",
            "summary": {
                "page_count": global_page_count,
                "line_count": global_line_count,
                "word_count": global_word_count,
                "confidence": global_dist
            },
            "pages": pages_metrics
        }

    return {
        "source_format": "unknown",
        "summary": {
            "page_count": 0, "line_count": 0, "word_count": 0,
            "confidence": confidence_distribution([])
        },
        "pages": []
    }


def build_image_quality_estimate(blob_bytes: bytes, result_json: dict, text: str, ocr_data: dict):
    """
    Calcule un score de qualité de l'IMAGE source (0-100).
    Pénalise : faible confiance OCR, petit fichier, texte court, forte inclinaison.
    """
    score = 100
    reasons = []
    size_bytes = len(blob_bytes)
    conf       = (((ocr_data or {}).get("summary") or {}).get("confidence") or {})
    avg_conf   = conf.get("avg")
    p10        = conf.get("p10")
    low_ratio  = ((conf.get("low_lt_0_60") or {}).get("ratio"))
    word_count = ((ocr_data.get("summary") or {}).get("word_count")) or 0
    line_count = ((ocr_data.get("summary") or {}).get("line_count")) or 0
    page_count = ((ocr_data.get("summary") or {}).get("page_count")) or 0

    if avg_conf is None:
        score -= 25
        reasons.append("Aucun score de confiance mot à mot disponible")
    else:
        if avg_conf < 0.50:   score -= 40; reasons.append("Confiance OCR moyenne très faible")
        elif avg_conf < 0.70: score -= 25; reasons.append("Confiance OCR moyenne faible")
        elif avg_conf < 0.85: score -= 10; reasons.append("Confiance OCR moyenne correcte mais perfectible")

    if p10 is not None:
        if p10 < 0.30:   score -= 20; reasons.append("Les mots les plus fragiles ont une confiance très basse")
        elif p10 < 0.50: score -= 10; reasons.append("Une partie des mots a une confiance basse")

    if low_ratio is not None:
        if low_ratio > 0.50:   score -= 25; reasons.append("Plus de la moitié des mots sont à faible confiance")
        elif low_ratio > 0.30: score -= 15; reasons.append("Beaucoup de mots sont à faible confiance")
        elif low_ratio > 0.15: score -= 5;  reasons.append("Une minorité notable de mots est à faible confiance")

    if size_bytes < 15 * 1024:   score -= 10; reasons.append("Fichier très léger")
    elif size_bytes < 40 * 1024: score -= 5;  reasons.append("Fichier assez léger")

    if len((text or "").strip()) < 20:   score -= 20; reasons.append("Très peu de texte extrait")
    elif len((text or "").strip()) < 80: score -= 8;  reasons.append("Texte extrait court")

    if word_count < 5:   score -= 20; reasons.append("Très peu de mots détectés")
    elif word_count < 20: score -= 8; reasons.append("Peu de mots détectés")

    if line_count == 0 and page_count > 0:
        score -= 25; reasons.append("Pages détectées sans lignes exploitables")

    page_angles = []
    for pg in ocr_data.get("pages", []):
        angle = ((pg.get("layout") or {}).get("angle"))
        if isinstance(angle, (int, float)):
            page_angles.append(abs(float(angle)))

    if page_angles:
        max_angle = max(page_angles)
        if max_angle > 20:  score -= 20; reasons.append("Document fortement incliné")
        elif max_angle > 8: score -= 8;  reasons.append("Document légèrement incliné")

    score = max(0, min(100, int(round(score))))
    if score >= 90:   label = "excellente"
    elif score >= 75: label = "bonne"
    elif score >= 55: label = "moyenne"
    elif score >= 35: label = "faible"
    else:             label = "très faible"

    return {
        "score": score, "label": label,
        "signals": {
            "file_size_bytes": size_bytes,
            "text_length": len(text or ""),
            "word_count": word_count, "line_count": line_count, "page_count": page_count,
            "avg_confidence": avg_conf, "p10_confidence": p10,
            "low_confidence_ratio_lt_0_60": low_ratio,
            "max_abs_angle": safe_round(max(page_angles)) if page_angles else None
        },
        "reasons": reasons
    }


def build_document_quality_estimate(text: str, lines_local: list, kv_local: dict, ocr_data: dict):
    """
    Calcule un score de qualité du DOCUMENT extrait (0-100).
    Pénalise : texte court, peu de lignes, confiance OCR hétérogène.
    Bonus si paires clé-valeur détectées (≥3).
    """
    score = 100
    reasons = []
    conf       = (((ocr_data or {}).get("summary") or {}).get("confidence") or {})
    avg_conf   = conf.get("avg")
    stddev     = conf.get("stddev")
    low_ratio  = ((conf.get("low_lt_0_60") or {}).get("ratio"))
    high_ratio = ((conf.get("high_gte_0_85") or {}).get("ratio"))
    text       = text or ""
    lines_local = lines_local or []
    kv_local    = kv_local or {}
    line_count  = len(lines_local)
    char_count  = len(text)
    token_count = len(re.findall(r"[^\s]+", text))
    kv_count    = len(kv_local)

    if avg_conf is not None:
        if avg_conf < 0.60:   score -= 30; reasons.append("Confiance OCR globale faible")
        elif avg_conf < 0.80: score -= 12; reasons.append("Confiance OCR globale moyenne")

    if low_ratio is not None and low_ratio > 0.30:
        score -= 12; reasons.append("Part importante de mots peu fiables")

    if high_ratio is not None and high_ratio < 0.40:
        score -= 8; reasons.append("Proportion limitée de mots très fiables")

    if stddev is not None and stddev > 0.20:
        score -= 6; reasons.append("Confiance OCR hétérogène")

    if char_count < 20:   score -= 25; reasons.append("Texte trop court pour exploitation fiable")
    elif char_count < 100: score -= 10; reasons.append("Texte assez court")

    if line_count < 2:   score -= 12; reasons.append("Très peu de lignes exploitables")
    if token_count < 5:  score -= 15; reasons.append("Très peu de tokens exploitables")

    if kv_count == 0 and token_count > 10:
        score -= 4; reasons.append("Aucune paire clé-valeur extraite")
    elif kv_count >= 3:
        score += 3  # ← Bonus : document structuré

    score = max(0, min(100, int(round(score))))
    if score >= 90:   label = "excellente"
    elif score >= 75: label = "bonne"
    elif score >= 55: label = "moyenne"
    elif score >= 35: label = "faible"
    else:             label = "très faible"

    return {
        "score": score, "label": label,
        "signals": {
            "char_count": char_count, "line_count": line_count,
            "token_count": token_count, "kv_count": kv_count,
            "avg_confidence": avg_conf, "stddev_confidence": stddev,
            "low_confidence_ratio_lt_0_60": low_ratio,
            "high_confidence_ratio_gte_0_85": high_ratio
        },
        "reasons": reasons
    }


def compute_global_document_score(ocr_metrics: dict, text_full: str, lines_local: list):
    """
    Score composite global (0-100) = 60% confiance + 25% couverture + 15% stabilité.
    Ce score synthétique est le plus utile pour décider si un document est exploitable.
    """
    summary   = (ocr_metrics or {}).get("summary") or {}
    conf      = summary.get("confidence") or {}
    pages     = (ocr_metrics or {}).get("pages") or []
    avg_conf  = conf.get("avg")
    p10_conf  = conf.get("p10")
    stddev_conf = conf.get("stddev")
    low_ratio = ((conf.get("low_lt_0_60") or {}).get("ratio"))
    page_count = summary.get("page_count") or 0
    word_count = summary.get("word_count") or 0
    line_count = summary.get("line_count") or len(lines_local or [])
    pages_with_text = sum(1 for p in pages if (p.get("word_count") or 0) > 0)

    avg_conf    = float(avg_conf)    if isinstance(avg_conf,    (int, float)) else 0.0
    p10_conf    = float(p10_conf)    if isinstance(p10_conf,    (int, float)) else 0.0
    stddev_conf = float(stddev_conf) if isinstance(stddev_conf, (int, float)) else 0.25
    low_ratio   = float(low_ratio)   if isinstance(low_ratio,   (int, float)) else 1.0

    # Score confiance (60% du total)
    score_confiance = 100 * (0.65 * avg_conf + 0.20 * p10_conf + 0.15 * (1 - low_ratio))

    # Score couverture (25% du total)
    pages_ratio   = (pages_with_text / page_count) if page_count > 0 else 0.0
    word_coverage = min(word_count / 200.0, 1.0)  # ← Seuil de saturation : 200 mots
    line_coverage = min(line_count / 50.0, 1.0)   # ← Seuil de saturation : 50 lignes
    score_couverture = 100 * (0.50 * pages_ratio + 0.30 * word_coverage + 0.20 * line_coverage)

    # Score stabilité (15% du total)
    stability_from_stddev = 1 - min(stddev_conf / 0.25, 1.0)
    page_avgs = [
        float(p.get("confidence", {}).get("avg"))
        for p in pages if isinstance(p.get("confidence", {}).get("avg"), (int, float))
    ]
    page_consistency = (1 - min((max(page_avgs) - min(page_avgs)) / 0.5, 1.0)) if len(page_avgs) >= 2 else 1.0
    score_stabilite = 100 * (0.70 * stability_from_stddev + 0.30 * page_consistency)

    score_global = max(0, min(100, round(
        0.60 * score_confiance + 0.25 * score_couverture + 0.15 * score_stabilite, 2
    )))

    if score_global >= 90:   label = "excellent"
    elif score_global >= 75: label = "bon"
    elif score_global >= 55: label = "moyen"
    elif score_global >= 35: label = "faible"
    else:                    label = "très faible"

    return {
        "score_global": score_global, "label": label,
        "detail": {
            "score_confiance":  round(score_confiance,  2),
            "score_couverture": round(score_couverture, 2),
            "score_stabilite":  round(score_stabilite,  2)
        },
        "signals": {
            "avg_confidence": avg_conf, "p10_confidence": p10_conf,
            "stddev_confidence": stddev_conf,
            "low_confidence_ratio_lt_0_60": low_ratio,
            "page_count": page_count, "pages_with_text": pages_with_text,
            "word_count": word_count, "line_count": line_count,
            "text_length": len(text_full or "")
        }
    }


def build_processing_diagnostics(result_json, text, lines_local, tokens_local, ocr_data):
    """
    Construit un rapport de diagnostics techniques pour le traitement OCR.
    Utile pour déboguer ou auditer la qualité de traitement page par page.
    """
    conf  = (((ocr_data or {}).get("summary") or {}).get("confidence") or {})
    pages = ocr_data.get("pages", []) or []

    page_diagnostics = []
    for p in pages:
        c = p.get("confidence") or {}
        page_diagnostics.append({
            "page_index":  p.get("page_index"),
            "word_count":  p.get("word_count"),
            "line_count":  p.get("line_count"),
            "avg_confidence": c.get("avg"),
            "low_confidence_ratio_lt_0_60":    ((c.get("low_lt_0_60") or {}).get("ratio")),
            "very_high_confidence_ratio_gte_0_95": ((c.get("very_high_gte_0_95") or {}).get("ratio")),
            "angle":  ((p.get("layout") or {}).get("angle")),
            "width":  ((p.get("layout") or {}).get("width")),
            "height": ((p.get("layout") or {}).get("height")),
            "unit":   ((p.get("layout") or {}).get("unit")),
        })

    return {
        "text_presence":           bool((text or "").strip()),
        "line_count":              len(lines_local or []),
        "token_count":             len(tokens_local or []),
        "page_count":              ((ocr_data.get("summary") or {}).get("page_count")) or 0,
        "word_count":              ((ocr_data.get("summary") or {}).get("word_count")) or 0,
        "has_confidence_scores":   conf.get("count", 0) > 0,
        "confidence_score_count":  conf.get("count", 0),
        "page_diagnostics":        page_diagnostics,
    }


# =============================================================
# OUTIL MCP PRINCIPAL — C'est lui que Dust appellera
# =============================================================
@mcp.tool()
def read_document_ocr(file_url: str, language: str = "auto") -> str:
    """
    Lit un document depuis une URL publique via Azure Computer Vision OCR.
    
    Extrait le texte, détecte les paires clé-valeur et calcule des scores
    de qualité pour évaluer la fiabilité du document analysé.
    
    Args:
        file_url: URL publique du fichier à analyser (JPG, PNG, PDF, TIFF, BMP...).
                  L'URL doit être directement accessible sans authentification.
        language: Code de langue ISO pour l'OCR.
                  Exemples : 'fr' (français), 'en' (anglais), 'nl' (néerlandais).
                  Utilisez 'auto' pour la détection automatique (par défaut).
    
    Returns:
        JSON string contenant :
        - text_full : texte brut complet extrait
        - pages : texte structuré page par page
        - information_extraite : paires clé-valeur détectées
        - global_document_score : score composite de qualité (0-100)
        - image_quality_estimate : qualité de l'image source
        - document_quality_estimate : exploitabilité du contenu
        - meta : métadonnées (taille, type, nb caractères...)
    """
    # ── Vérification des credentials Azure ───────────────────
    if not AZURE_VISION_ENDPOINT or not AZURE_VISION_KEY:
        return json.dumps({
            "error": (
                "Credentials Azure non configurés sur le serveur. "
                "Vérifiez les variables d'environnement AZURE_VISION_ENDPOINT "
                "et AZURE_VISION_KEY sur Railway."
            )
        }, ensure_ascii=False)

    try:
        # ── 1. Téléchargement du fichier ──────────────────────
        blob, content_type, ext = download_file_from_url(file_url.strip())

        # ── 2. Analyse OCR Azure ──────────────────────────────
        result = azure_read_analyze(
            blob, AZURE_VISION_ENDPOINT, AZURE_VISION_KEY, language.strip()
        )

        # ── 3. Extraction du texte ────────────────────────────
        text_full, pages_struct = extract_text_from_azure_result(result)

        # ── 4. Construction de la liste de lignes ─────────────
        lines = []
        if pages_struct:
            for pg in pages_struct:
                lines.extend(pg["lines"])
        else:
            lines = [l.strip() for l in (text_full or "").splitlines() if l and l.strip()]

        # ── 5. Extraction des paires clé-valeur ───────────────
        kv = {}
        for ln in lines:
            l = ln.strip()
            matched = False
            for sep in [":", "="]:
                if sep in l:
                    left, right = l.split(sep, 1)
                    left, right = left.strip(), right.strip()
                    if 1 < len(left) < 80 and not left.isdigit():
                        k = normalize_key(left)
                        v = normalize_val(right)
                        if k and v:
                            kv[k] = v if k not in kv else (kv[k] + " | " + v if kv[k] != v else kv[k])
                    matched = True
                    break
            if matched:
                continue
            for sep in [" - ", " – ", " — "]:
                if sep in l:
                    left, right = l.split(sep, 1)
                    left, right = left.strip(), right.strip()
                    if 1 < len(left) < 80 and not left.isdigit():
                        k = normalize_key(left)
                        v = normalize_val(right)
                        if k and v:
                            kv[k] = v if k not in kv else (kv[k] + " | " + v if kv[k] != v else kv[k])
                    break

        # ── 6. Extraction des tokens (max 2000) ───────────────
        tokens = []
        for ln in lines:
            tokens.extend(re.findall(r"[^\s]+", ln))
        tokens = tokens[:2000]

        # ── 7. Calcul des métriques et scores ─────────────────
        ocr_metrics               = extract_ocr_metrics(result)
        image_quality_estimate    = build_image_quality_estimate(blob, result, text_full, ocr_metrics)
        document_quality_estimate = build_document_quality_estimate(text_full, lines, kv, ocr_metrics)
        global_document_score     = compute_global_document_score(ocr_metrics, text_full, lines)
        processing_diagnostics    = build_processing_diagnostics(result, text_full, lines, tokens, ocr_metrics)

        # ── 8. Construction du résultat final ─────────────────
        output = {
            "file_url": file_url,
            "meta": {
                "content_type":      content_type,
                "guessed_extension": ext,
                "file_size_bytes":   len(blob),
                "char_count":        len(text_full),
                "line_count":        len(lines),
                "token_count":       len(tokens),
                "azure_status":      result.get("status"),
            },
            "text_full":            text_full,
            "pages":                pages_struct,
            "lines":                lines,
            "information_extraite": kv,
            "tokens":               tokens,
            "ocr_scores": {
                "source_format": ocr_metrics.get("source_format"),
                "summary":       ocr_metrics.get("summary"),
                "pages":         ocr_metrics.get("pages"),
            },
            "image_quality_estimate":    image_quality_estimate,
            "document_quality_estimate": document_quality_estimate,
            "global_document_score":     global_document_score,
            "processing_diagnostics":    processing_diagnostics,
        }

        return json.dumps(output, ensure_ascii=False, indent=2)
        # ↑ ensure_ascii=False : préserve les caractères accentués (é, à, ô...)
        # indent=2 : JSON lisible pour le débogage

    except Exception as e:
        # ← Capture toute erreur inattendue et la renvoie proprement à Dust
        return json.dumps({
            "error":    str(e),
            "file_url": file_url
        }, ensure_ascii=False)


# =============================================================
# DÉMARRAGE DU SERVEUR HTTP
# =============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # ↑ Railway injecte automatiquement la variable PORT.
    # On utilise 8000 comme valeur par défaut pour les tests locaux.

    print(f"🚀 Démarrage du serveur MCP Azure OCR sur le port {port}")
    print(f"📡 Transport : streamable-http")
    print(f"🔗 URL MCP   : http://0.0.0.0:{port}/mcp")

    mcp.run(
        transport="streamable-http",
        # ↑ Transport HTTP streaming — le standard MCP pour les serveurs distants.
        # C'est ce que Dust utilise pour communiquer avec votre serveur.
        host="0.0.0.0",
        # ↑ Écoute sur toutes les interfaces réseau (requis sur Railway)
        # Ne jamais utiliser "localhost" ou "127.0.0.1" en production !
        port=port,
    )