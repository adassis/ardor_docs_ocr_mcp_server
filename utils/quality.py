# =============================================================
# utils/quality.py — Scores de qualité OCR
# =============================================================
# Fonctions de calcul des métriques et scores qualité.
# Isolées ici car volumineuses et indépendantes du reste.
# Importées uniquement par les outils qui exposent ces scores.
# =============================================================


def safe_round(v, digits=4):
    """
    Arrondit un nombre flottant en toute sécurité.
    Retourne None si la valeur n'est pas numérique.
    """
    if isinstance(v, (int, float)):
        return round(float(v), digits)
    return None


def percentile(sorted_values, p):
    """
    Calcule le p-ième percentile d'une liste de valeurs déjà triées.

    Args:
        sorted_values : liste de nombres triés par ordre croissant
        p             : percentile entre 0 et 1 (ex: 0.50 = médiane)

    Returns:
        float ou None si la liste est vide
    """
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
    Calcule une distribution statistique complète des scores de confiance OCR.

    Chaque mot reconnu par Azure est associé à un score de confiance entre 0 et 1.
    Cette fonction agrège ces scores en statistiques exploitables.

    Args:
        confidences : liste de floats entre 0.0 et 1.0

    Returns:
        dict avec : count, avg, min, max, median, p10, p25, p75, p90,
                    stddev, et 5 buckets de répartition par seuil
    """
    if not confidences:
        return {
            "count": 0, "avg": None, "min": None, "max": None,
            "median": None, "p10": None, "p25": None, "p75": None, "p90": None,
            "stddev": None,
            "low_lt_0_50":        {"count": 0, "ratio": None},
            "low_lt_0_60":        {"count": 0, "ratio": None},
            "medium_0_60_0_85":   {"count": 0, "ratio": None},
            "high_gte_0_85":      {"count": 0, "ratio": None},
            "very_high_gte_0_95": {"count": 0, "ratio": None},
        }

    vals = sorted([float(x) for x in confidences])
    count = len(vals)
    avg = sum(vals) / count
    variance = sum((x - avg) ** 2 for x in vals) / count
    stddev = variance ** 0.5

    c_lt_050 = sum(1 for x in vals if x < 0.50)
    c_lt_060 = sum(1 for x in vals if x < 0.60)
    c_mid    = sum(1 for x in vals if 0.60 <= x < 0.85)
    c_hi     = sum(1 for x in vals if x >= 0.85)
    c_vhi    = sum(1 for x in vals if x >= 0.95)

    return {
        "count":  count,
        "avg":    safe_round(avg),
        "min":    safe_round(vals[0]),
        "max":    safe_round(vals[-1]),
        "median": safe_round(percentile(vals, 0.50)),
        "p10":    safe_round(percentile(vals, 0.10)),
        "p25":    safe_round(percentile(vals, 0.25)),
        "p75":    safe_round(percentile(vals, 0.75)),
        "p90":    safe_round(percentile(vals, 0.90)),
        "stddev": safe_round(stddev),
        "low_lt_0_50":        {"count": c_lt_050, "ratio": safe_round(c_lt_050 / count)},
        "low_lt_0_60":        {"count": c_lt_060, "ratio": safe_round(c_lt_060 / count)},
        "medium_0_60_0_85":   {"count": c_mid,    "ratio": safe_round(c_mid    / count)},
        "high_gte_0_85":      {"count": c_hi,     "ratio": safe_round(c_hi     / count)},
        "very_high_gte_0_95": {"count": c_vhi,    "ratio": safe_round(c_vhi    / count)},
    }


def extract_ocr_metrics(result_json: dict) -> dict:
    """
    Extrait les métriques détaillées de la réponse Azure OCR.
    Gère les 2 formats de réponse (readResults v3.2 et pages v4+).

    Returns:
        dict avec source_format, summary global, et détail par page
    """
    ar = result_json.get("analyzeResult", {})
    global_word_confidences = []
    pages_metrics = []
    global_word_count = 0
    global_line_count = 0
    global_page_count = 0

    # ── Format v3.2 ───────────────────────────────────────────
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
                },
            })
        return {
            "source_format": "readResults",
            "summary": {
                "page_count": global_page_count,
                "line_count": global_line_count,
                "word_count": global_word_count,
                "confidence": confidence_distribution(global_word_confidences),
            },
            "pages": pages_metrics,
        }

    # ── Format v4+ ────────────────────────────────────────────
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
                        page_words.extend(__import__("re").findall(r"[^\s]+", txt))
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
                },
            })
        return {
            "source_format": "pages",
            "summary": {
                "page_count": global_page_count,
                "line_count": global_line_count,
                "word_count": global_word_count,
                "confidence": confidence_distribution(global_word_confidences),
            },
            "pages": pages_metrics,
        }

    return {
        "source_format": "unknown",
        "summary": {
            "page_count": 0, "line_count": 0, "word_count": 0,
            "confidence": confidence_distribution([]),
        },
        "pages": [],
    }


def build_image_quality_estimate(blob_bytes: bytes, result_json: dict, text: str, ocr_data: dict) -> dict:
    """
    Calcule un score qualité de l'IMAGE source (0-100).
    Pénalise : faible confiance OCR, fichier trop léger, peu de texte, document incliné.
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
        score -= 25; reasons.append("Aucun score de confiance mot à mot disponible")
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

    if word_count < 5:    score -= 20; reasons.append("Très peu de mots détectés")
    elif word_count < 20: score -= 8;  reasons.append("Peu de mots détectés")

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
            "text_length":     len(text or ""),
            "word_count":      word_count,
            "line_count":      line_count,
            "page_count":      page_count,
            "avg_confidence":                  avg_conf,
            "p10_confidence":                  p10,
            "low_confidence_ratio_lt_0_60":    low_ratio,
            "max_abs_angle": safe_round(max(page_angles)) if page_angles else None,
        },
        "reasons": reasons,
    }


def build_document_quality_estimate(text: str, lines_local: list, kv_local: dict, ocr_data: dict) -> dict:
    """
    Calcule un score qualité du DOCUMENT extrait (0-100).
    Évalue la richesse et la fiabilité du contenu textuel obtenu.
    """
    import re
    score = 100
    reasons = []
    conf       = (((ocr_data or {}).get("summary") or {}).get("confidence") or {})
    avg_conf   = conf.get("avg")
    stddev     = conf.get("stddev")
    low_ratio  = ((conf.get("low_lt_0_60") or {}).get("ratio"))
    high_ratio = ((conf.get("high_gte_0_85") or {}).get("ratio"))
    text        = text or ""
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

    if char_count < 20:    score -= 25; reasons.append("Texte trop court pour exploitation fiable")
    elif char_count < 100: score -= 10; reasons.append("Texte assez court")

    if line_count < 2:   score -= 12; reasons.append("Très peu de lignes exploitables")
    if token_count < 5:  score -= 15; reasons.append("Très peu de tokens exploitables")

    if kv_count == 0 and token_count > 10:
        score -= 4; reasons.append("Aucune paire clé-valeur extraite")
    elif kv_count >= 3:
        score += 3

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
            "avg_confidence":                  avg_conf,
            "stddev_confidence":               stddev,
            "low_confidence_ratio_lt_0_60":    low_ratio,
            "high_confidence_ratio_gte_0_85":  high_ratio,
        },
        "reasons": reasons,
    }


def compute_global_document_score(ocr_metrics: dict, text_full: str, lines_local: list) -> dict:
    """
    Score global pondéré (0-100) combinant :
    - Score de confiance  (60%) : qualité de la reconnaissance mot à mot
    - Score de couverture (25%) : proportion du document couvert par l'OCR
    - Score de stabilité  (15%) : homogénéité entre les pages
    """
    summary     = (ocr_metrics or {}).get("summary") or {}
    conf        = summary.get("confidence") or {}
    pages       = (ocr_metrics or {}).get("pages") or []
    avg_conf    = conf.get("avg")
    p10_conf    = conf.get("p10")
    stddev_conf = conf.get("stddev")
    low_ratio   = ((conf.get("low_lt_0_60") or {}).get("ratio"))
    page_count  = summary.get("page_count") or 0
    word_count  = summary.get("word_count") or 0
    line_count  = summary.get("line_count") or len(lines_local or [])
    pages_with_text = sum(1 for p in pages if (p.get("word_count") or 0) > 0)

    avg_conf    = float(avg_conf)    if isinstance(avg_conf,    (int, float)) else 0.0
    p10_conf    = float(p10_conf)    if isinstance(p10_conf,    (int, float)) else 0.0
    stddev_conf = float(stddev_conf) if isinstance(stddev_conf, (int, float)) else 0.25
    low_ratio   = float(low_ratio)   if isinstance(low_ratio,   (int, float)) else 1.0

    score_confiance  = 100 * (0.65 * avg_conf + 0.20 * p10_conf + 0.15 * (1 - low_ratio))
    pages_ratio      = (pages_with_text / page_count) if page_count > 0 else 0.0
    word_coverage    = min(word_count / 200.0, 1.0)
    line_coverage    = min(line_count / 50.0, 1.0)
    score_couverture = 100 * (0.50 * pages_ratio + 0.30 * word_coverage + 0.20 * line_coverage)
    stability_from_stddev = 1 - min(stddev_conf / 0.25, 1.0)
    page_avgs = [
        float(p.get("confidence", {}).get("avg"))
        for p in pages if isinstance(p.get("confidence", {}).get("avg"), (int, float))
    ]
    page_consistency = (
        1 - min((max(page_avgs) - min(page_avgs)) / 0.5, 1.0)
    ) if len(page_avgs) >= 2 else 1.0
    score_stabilite  = 100 * (0.70 * stability_from_stddev + 0.30 * page_consistency)
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
            "score_confiance":  round(score_confiance, 2),
            "score_couverture": round(score_couverture, 2),
            "score_stabilite":  round(score_stabilite, 2),
        },
        "signals": {
            "avg_confidence": avg_conf, "p10_confidence": p10_conf,
            "stddev_confidence": stddev_conf,
            "low_confidence_ratio_lt_0_60": low_ratio,
            "page_count": page_count, "pages_with_text": pages_with_text,
            "word_count": word_count, "line_count": line_count,
            "text_length": len(text_full or ""),
        },
    }


def build_processing_diagnostics(result_json, text, lines_local, tokens_local, ocr_data) -> dict:
    """
    Construit un rapport de diagnostic détaillé par page.
    Utile pour détecter les pages problématiques (inclinées, vides, peu fiables).
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
            "avg_confidence":                          c.get("avg"),
            "low_confidence_ratio_lt_0_60":            ((c.get("low_lt_0_60")        or {}).get("ratio")),
            "very_high_confidence_ratio_gte_0_95":     ((c.get("very_high_gte_0_95") or {}).get("ratio")),
            "angle":  ((p.get("layout") or {}).get("angle")),
            "width":  ((p.get("layout") or {}).get("width")),
            "height": ((p.get("layout") or {}).get("height")),
            "unit":   ((p.get("layout") or {}).get("unit")),
        })
    return {
        "text_presence":          bool((text or "").strip()),
        "line_count":             len(lines_local or []),
        "token_count":            len(tokens_local or []),
        "page_count":             ((ocr_data.get("summary") or {}).get("page_count")) or 0,
        "word_count":             ((ocr_data.get("summary") or {}).get("word_count")) or 0,
        "has_confidence_scores":  conf.get("count", 0) > 0,
        "confidence_score_count": conf.get("count", 0),
        "page_diagnostics":       page_diagnostics,
    }