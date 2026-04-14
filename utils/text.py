# =============================================================
# utils/text.py — Fonctions de traitement du texte OCR
# =============================================================
# Nettoyage, normalisation, extraction de paires clé-valeur, tokenisation.
# Importé par tous les outils qui manipulent du texte extrait par OCR.
# Aucune dépendance vers d'autres modules internes du projet.
# =============================================================

import re


def normalize_key(k: str) -> str:
    """
    Nettoie une clé pour la stocker dans le dict clé-valeur.

    Transformations :
    - Supprime les espaces en début/fin
    - Supprime les : - = parasites en bordure
    - Réduit les espaces multiples à un seul
    - Met en minuscules pour uniformiser

    Ex: "  Nom :  " → "nom"
        "DATE=" → "date"
    """
    k = k.strip().strip(":").strip("-").strip("=").strip()
    k = re.sub(r"\s+", " ", k)
    return k.lower()


def normalize_val(v: str) -> str:
    """
    Nettoie une valeur pour la stocker dans le dict clé-valeur.

    Transformations :
    - Supprime les espaces en début/fin
    - Réduit les espaces multiples à un seul
    - Conserve la casse originale (contrairement aux clés)

    Ex: "  Dupont  Jean  " → "Dupont Jean"
    """
    v = v.strip()
    v = re.sub(r"\s+", " ", v)
    return v


def extract_kv_from_lines(lines: list) -> dict:
    """
    Extrait les paires clé-valeur depuis une liste de lignes de texte OCR.

    Détecte les séparateurs courants dans les documents :
    - "Nom : Dupont"          → séparateur ":"
    - "Montant = 1500"        → séparateur "="
    - "Référence - ABC123"    → séparateur " - "
    - "Date — 01/01/2025"     → séparateur " — " (tiret long)
    - "Client – Entreprise"   → séparateur " – " (tiret moyen)

    Règles de filtrage pour éviter le bruit :
    - La clé doit faire entre 2 et 79 caractères
    - La clé ne doit pas être un nombre pur (ex: "1 : 2" est ignoré)

    Gestion des doublons :
    - Si une clé apparaît 2 fois avec la même valeur → on garde une seule fois
    - Si une clé apparaît 2 fois avec des valeurs différentes → on concatène avec " | "

    Args:
        lines : liste de chaînes de texte (sorties d'Azure OCR)

    Returns:
        dict : {"nom": "Dupont", "date": "01/01/2025", ...}
    """
    kv = {}

    for ln in lines:
        l = ln.strip()
        matched = False

        # ── Tentative avec séparateurs : et = ─────────────────
        for sep in [":", "="]:
            if sep in l:
                left, right = l.split(sep, 1)
                left, right = left.strip(), right.strip()
                if 1 < len(left) < 80 and not left.isdigit():
                    k = normalize_key(left)
                    v = normalize_val(right)
                    if k and v:
                        if k not in kv:
                            kv[k] = v
                        elif kv[k] != v:
                            kv[k] = kv[k] + " | " + v
                        # Si kv[k] == v : doublon exact, on ne fait rien
                matched = True
                break

        if matched:
            continue

        # ── Tentative avec séparateurs tirets ─────────────────
        for sep in [" - ", " – ", " — "]:
            if sep in l:
                left, right = l.split(sep, 1)
                left, right = left.strip(), right.strip()
                if 1 < len(left) < 80 and not left.isdigit():
                    k = normalize_key(left)
                    v = normalize_val(right)
                    if k and v:
                        if k not in kv:
                            kv[k] = v
                        elif kv[k] != v:
                            kv[k] = kv[k] + " | " + v
                break

    return kv


def extract_tokens(lines: list, max_tokens: int = 2000) -> list:
    """
    Extrait tous les mots individuels depuis une liste de lignes.

    Utile pour des analyses lexicales ou des recherches de mots-clés.
    Limité à max_tokens pour éviter des réponses JSON trop volumineuses.

    Args:
        lines      : liste de chaînes de texte
        max_tokens : nombre maximum de tokens à retourner (défaut : 2000)

    Returns:
        list : ["mot1", "mot2", "mot3", ...]
    """
    tokens = []
    for ln in lines:
        # re.findall(r"[^\s]+", ln) : tout ce qui n'est pas un espace
        # = chaque "mot" (ponctuation incluse : "Dupont," est un token)
        tokens.extend(re.findall(r"[^\s]+", ln))
    return tokens[:max_tokens]