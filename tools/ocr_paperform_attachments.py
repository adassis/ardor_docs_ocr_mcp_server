# =============================================================
# tools/ocr_paperform_attachments.py — OCR pièces jointes Paperform
# =============================================================
# Expose 3 outils :
#   - debug_paperform_submission         : inspecte la réponse brute
#   - read_paperform_submission_ocr      : OCR d'une soumission
#   - read_paperform_submission_ocr_bulk : OCR de plusieurs soumissions
#
# Structure réelle de l'API Paperform :
#   results.submission.data.{field_key} = valeur
#   Les fichiers = listes d'objets [{url, name, type, size}]
# =============================================================

import json
import requests

from config import (
    AZURE_VISION_ENDPOINT,
    AZURE_VISION_KEY,
    PAPERFORM_API_TOKEN,
)
from utils.azure import azure_read_analyze, extract_text_from_azure_result
from utils.text import extract_kv_from_lines, extract_tokens


# =============================================================
# FONCTIONS PRIVÉES
# =============================================================

def _get_paperform_submission(submission_id: str) -> dict:
    """
    Récupère une soumission Paperform via son ID.

    Endpoint : GET https://api.paperform.co/v1/submissions/{id}
    Auth     : Bearer token dans le header Authorization.

    Returns:
        dict : réponse JSON complète de l'API Paperform

    Raises:
        RuntimeError : si token manquant, 401, 404 ou autre erreur HTTP
    """
    if not PAPERFORM_API_TOKEN:
        raise RuntimeError(
            "PAPERFORM_API_TOKEN non configuré. "
            "Générez un token sur paperform.co/account/developer "
            "et ajoutez-le dans les variables Railway."
        )

    headers = {
        "Authorization": f"Bearer {PAPERFORM_API_TOKEN}",
        "Accept":        "application/json",
    }

    r = requests.get(
        f"https://api.paperform.co/v1/submissions/{submission_id}",
        headers=headers,
        timeout=30
    )

    if r.status_code == 401:
        raise RuntimeError("PAPERFORM_API_TOKEN invalide ou expiré.")
    if r.status_code == 404:
        raise RuntimeError(f"Soumission '{submission_id}' introuvable.")
    if not r.ok:
        raise RuntimeError(f"Paperform API HTTP {r.status_code}: {r.text[:400]}")

    return r.json()


def _extract_file_urls(submission_data: dict) -> list:
    """
    Extrait toutes les URLs de fichiers d'une soumission Paperform.

    Structure réelle de l'API :
        results.submission.data.{field_key} = valeur

    Les champs fichiers sont des listes d'objets :
        [{"url": "...", "name": "...", "type": "image/jpeg", "size": 123456}]

    On les détecte en cherchant les valeurs qui sont :
    - une liste non vide
    - dont le premier élément est un dict avec une clé "url"

    Args:
        submission_data : réponse JSON complète de l'API Paperform

    Returns:
        list : [{url, field_key, filename, mime_type, size}]
    """
    results    = submission_data.get("results") or {}
    submission = results.get("submission") or {}
    data       = submission.get("data") or {}

    files = []

    for field_key, value in data.items():
        # On cherche uniquement les listes non vides
        if not isinstance(value, list) or not value:
            continue

        # Le premier élément doit être un objet avec une clé "url"
        first = value[0]
        if not isinstance(first, dict) or "url" not in first:
            continue

        # C'est un champ fichier — on extrait chaque fichier
        for file_obj in value:
            url = file_obj.get("url") or ""
            if not url.startswith("http"):
                continue

            files.append({
                "url":       url,
                "field_key": field_key,
                "filename":  file_obj.get("name") or "",
                "mime_type": file_obj.get("type") or "",
                "size":      file_obj.get("size"),
            })

    return files


def _ocr_file_from_url(url: str, language: str) -> dict:
    """
    Télécharge un fichier depuis une URL Paperform et effectue l'OCR.

    Les URLs Paperform sont des URLs signées (avec expires + signature)
    accessibles directement sans authentification.

    Args:
        url      : URL signée du fichier (CDN Paperform / S3)
        language : code langue pour Azure OCR ('fr', 'en', 'auto'...)

    Returns:
        dict : {status, text_full, information_extraite, tokens, char_count, error}
    """
    r = requests.get(url, timeout=60, allow_redirects=True)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code} pour le téléchargement du fichier.")

    blob = r.content

    # OCR Azure
    result     = azure_read_analyze(blob, AZURE_VISION_ENDPOINT, AZURE_VISION_KEY, language)
    text_full, pages_struct = extract_text_from_azure_result(result)

    # Extraction des lignes de texte
    lines = []
    if pages_struct:
        for pg in pages_struct:
            lines.extend(pg["lines"])
    else:
        lines = [l.strip() for l in (text_full or "").splitlines() if l and l.strip()]

    kv     = extract_kv_from_lines(lines)
    tokens = extract_tokens(lines, max_tokens=2000)

    return {
        "status":               "success",
        "char_count":           len(text_full),
        "text_full":            text_full,
        "information_extraite": kv,
        "tokens":               tokens,
        "error":                None,
    }


# =============================================================
# ENREGISTREMENT DES OUTILS
# =============================================================

def register(mcp):

    @mcp.tool()
    def debug_paperform_submission(submission_id: str) -> str:
        """
        Retourne la réponse brute de l'API Paperform pour une soumission.
        Outil de debug — inspecte la structure exacte de la réponse.

        Args:
            submission_id : UUID de la soumission à inspecter

        Returns:
            JSON brut tel que retourné par l'API Paperform.
        """
        try:
            data = _get_paperform_submission(submission_id.strip())
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)


    @mcp.tool()
    def read_paperform_submission_ocr(
        submission_id: str,
        language: str = "auto"
    ) -> str:
        """
        Récupère une soumission Paperform et effectue l'OCR sur toutes
        les pièces jointes (photos, documents) qu'elle contient.

        Utilisez cet outil quand l'utilisateur fournit un ID de soumission
        Paperform et souhaite analyser les documents uploadés dans le formulaire.

        Args:
            submission_id : ID de la soumission Paperform.
                            Exemple : "69e08f0fd11d2bdd600acb9e"
            language      : Code langue ISO ('fr', 'en', 'nl'...).
                            'auto' pour détection automatique (défaut).

        Returns:
            JSON avec :
            - submission_id : identifiant de la soumission
            - form_id       : identifiant du formulaire
            - submitted_at  : date de soumission
            - total_files   : nombre de fichiers trouvés
            - files_ocr     : liste de résultats OCR par fichier
              {field_key, filename, mime_type, text_full, information_extraite}
        """
        if not AZURE_VISION_ENDPOINT or not AZURE_VISION_KEY:
            return json.dumps({"error": "Credentials Azure non configurés."}, ensure_ascii=False)

        if not PAPERFORM_API_TOKEN:
            return json.dumps({"error": "PAPERFORM_API_TOKEN non configuré."}, ensure_ascii=False)

        try:
            # ── 1. Récupère la soumission ──────────────────────
            submission_data = _get_paperform_submission(submission_id.strip())
            results         = submission_data.get("results") or {}
            submission      = results.get("submission") or {}

            form_id      = submission.get("form_id") or ""
            submitted_at = submission.get("created_at") or ""

            # ── 2. Extrait les URLs de fichiers ────────────────
            files = _extract_file_urls(submission_data)

            if not files:
                return json.dumps({
                    "submission_id": submission_id,
                    "form_id":       form_id,
                    "submitted_at":  submitted_at,
                    "total_files":   0,
                    "message":       "Aucune pièce jointe trouvée dans cette soumission.",
                    "files_ocr":     [],
                }, ensure_ascii=False, indent=2)

            # ── 3. OCR de chaque fichier ───────────────────────
            files_ocr = []
            for file_info in files:
                try:
                    ocr_result = _ocr_file_from_url(file_info["url"], language.strip())
                    files_ocr.append({
                        "field_key":  file_info["field_key"],
                        "filename":   file_info["filename"],
                        "mime_type":  file_info["mime_type"],
                        "size":       file_info["size"],
                        **ocr_result,
                    })
                except Exception as e:
                    files_ocr.append({
                        "field_key": file_info["field_key"],
                        "filename":  file_info["filename"],
                        "url":       file_info["url"],
                        "status":    "error",
                        "error":     str(e),
                    })

            output = {
                "submission_id": submission_id,
                "form_id":       form_id,
                "submitted_at":  submitted_at,
                "total_files":   len(files),
                "files_ocr":     files_ocr,
            }
            return json.dumps(output, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps(
                {"error": str(e), "submission_id": submission_id},
                ensure_ascii=False
            )


    @mcp.tool()
    def read_paperform_submission_ocr_bulk(
        submission_ids_csv: str,
        language: str = "auto"
    ) -> str:
        """
        OCR des pièces jointes de plusieurs soumissions Paperform en un seul appel.

        Utile quand plusieurs soumissions doivent être traitées à la suite
        (ex : toutes les soumissions du jour).

        Args:
            submission_ids_csv : IDs de soumissions séparés par des virgules (max 5).
                                 Exemple : "id1,id2,id3"
            language           : Code langue ISO. 'auto' pour détection automatique.

        Returns:
            JSON avec :
            - total   : nombre de soumissions soumises
            - success : nombre traitées avec succès
            - errors  : nombre d'échecs
            - results : liste de résultats par soumission
        """
        if not AZURE_VISION_ENDPOINT or not AZURE_VISION_KEY:
            return json.dumps({"error": "Credentials Azure non configurés."}, ensure_ascii=False)

        if not PAPERFORM_API_TOKEN:
            return json.dumps({"error": "PAPERFORM_API_TOKEN non configuré."}, ensure_ascii=False)

        ids = [sid.strip() for sid in submission_ids_csv.split(",") if sid.strip()]
        if not ids:
            return json.dumps({"error": "Aucun submission_id fourni."}, ensure_ascii=False)

        ids = ids[:5]  # Plafond à 5 — chaque soumission peut avoir plusieurs fichiers

        all_results   = []
        success_count = 0
        error_count   = 0

        for submission_id in ids:
            try:
                submission_data = _get_paperform_submission(submission_id)
                results         = submission_data.get("results") or {}
                submission      = results.get("submission") or {}

                form_id      = submission.get("form_id") or ""
                submitted_at = submission.get("created_at") or ""
                files        = _extract_file_urls(submission_data)

                files_ocr = []
                for file_info in files:
                    try:
                        ocr_result = _ocr_file_from_url(file_info["url"], language.strip())
                        files_ocr.append({
                            "field_key": file_info["field_key"],
                            "filename":  file_info["filename"],
                            "mime_type": file_info["mime_type"],
                            **ocr_result,
                        })
                    except Exception as e:
                        files_ocr.append({
                            "field_key": file_info["field_key"],
                            "filename":  file_info["filename"],
                            "status":    "error",
                            "error":     str(e),
                        })

                all_results.append({
                    "submission_id": submission_id,
                    "form_id":       form_id,
                    "submitted_at":  submitted_at,
                    "total_files":   len(files),
                    "status":        "success",
                    "files_ocr":     files_ocr,
                })
                success_count += 1

            except Exception as e:
                all_results.append({
                    "submission_id": submission_id,
                    "status":        "error",
                    "error":         str(e),
                })
                error_count += 1

        output = {
            "total":   len(ids),
            "success": success_count,
            "errors":  error_count,
            "results": all_results,
        }
        return json.dumps(output, ensure_ascii=False, indent=2)