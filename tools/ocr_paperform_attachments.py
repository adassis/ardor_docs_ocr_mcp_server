# =============================================================
# tools/ocr_paperform.py — Outils MCP : OCR pièces jointes Paperform
# =============================================================
# Expose 2 outils :
#   - read_paperform_submission_ocr      : OCR d'une soumission
#   - read_paperform_submission_ocr_bulk : OCR de plusieurs soumissions
#
# Workflow :
#   1. Appel API Paperform → récupère la soumission
#   2. Extraction des URLs de fichiers (champs type file/image)
#   3. Téléchargement de chaque fichier
#   4. OCR Azure sur chaque fichier
#   5. Retour des résultats
#
# Dépend de : config, utils/azure, utils/text
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
    Auth : Bearer token dans le header Authorization.

    Args:
        submission_id : identifiant UUID de la soumission

    Returns:
        dict : corps JSON de la réponse Paperform

    Raises:
        RuntimeError : si token manquant ou erreur HTTP
    """
    if not PAPERFORM_API_TOKEN:
        raise RuntimeError(
            "PAPERFORM_API_TOKEN non configuré. "
            "Générez un token sur paperform.co/account/developer "
            "et ajoutez-le dans les variables Railway."
        )

    headers = {
        "Authorization": f"Bearer {PAPERFORM_API_TOKEN}",
        "Accept": "application/json",
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

    Dans une soumission Paperform, les fichiers sont dans `answers` :
    Chaque `answer` a un `type` ("file", "image", "signature"...)
    et une `value` qui peut être :
      - une liste d'URLs : ["https://...file1.pdf", "https://...file2.jpg"]
      - une URL directe : "https://...file.pdf"
      - null si pas de fichier uploadé

    Args:
        submission_data : réponse JSON de l'API Paperform

    Returns:
        list : liste de dicts {url, field_title, field_key, field_type}
    """
    # La soumission est dans results.submission ou results selon l'API
    results     = submission_data.get("results") or {}
    answers     = results.get("answers") or []
    file_types  = {"file", "image", "signature", "upload"}
    files       = []

    for answer in answers:
        field_type  = (answer.get("type") or "").lower()
        field_key   = answer.get("id") or answer.get("key") or ""
        field_title = answer.get("title") or field_key
        value       = answer.get("value")

        # On ne traite que les champs de type fichier
        if field_type not in file_types and "file" not in field_type:
            continue

        if not value:
            continue  # Champ vide, pas de fichier uploadé

        # La value peut être une liste ou une URL directe
        urls = value if isinstance(value, list) else [value]

        for url in urls:
            if isinstance(url, str) and url.startswith("http"):
                files.append({
                    "url":         url,
                    "field_title": field_title,
                    "field_key":   field_key,
                    "field_type":  field_type,
                })

    return files


def _ocr_file_from_url(url: str, language: str) -> dict:
    """
    Télécharge un fichier depuis une URL Paperform et effectue l'OCR.

    Args:
        url      : URL du fichier (CDN Paperform)
        language : code langue pour Azure OCR

    Returns:
        dict : {status, text_full, information_extraite, char_count, error}
    """
    # Téléchargement direct — les URLs Paperform sont accessibles sans auth
    r = requests.get(url, timeout=60, allow_redirects=True)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code} pour {url}")

    blob = r.content

    # OCR Azure
    result     = azure_read_analyze(blob, AZURE_VISION_ENDPOINT, AZURE_VISION_KEY, language)
    text_full, pages_struct = extract_text_from_azure_result(result)

    # Extraction des lignes
    lines = []
    if pages_struct:
        for pg in pages_struct:
            lines.extend(pg["lines"])
    else:
        lines = [l.strip() for l in (text_full or "").splitlines() if l and l.strip()]

    kv     = extract_kv_from_lines(lines)
    tokens = extract_tokens(lines, max_tokens=2000)

    return {
        "status":              "success",
        "char_count":          len(text_full),
        "text_full":           text_full,
        "information_extraite": kv,
        "tokens":              tokens,
        "error":               None,
    }


# =============================================================
# ENREGISTREMENT DES OUTILS
# =============================================================

def register(mcp):

    @mcp.tool()
    def read_paperform_submission_ocr(
        submission_id: str,
        language: str = "auto"
    ) -> str:
        """
        Récupère une soumission Paperform et effectue l'OCR sur toutes
        les pièces jointes (champs file/image) qu'elle contient.

        Utilisez cet outil quand l'utilisateur fournit un ID de soumission
        Paperform et souhaite analyser les documents qui y sont attachés.

        Args:
            submission_id : UUID de la soumission Paperform.
                            Exemple : "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
            language      : Code langue ISO ('fr', 'en', 'nl'...).
                            'auto' pour détection automatique (défaut).

        Returns:
            JSON avec :
            - submission_id    : identifiant de la soumission
            - form_id          : identifiant du formulaire
            - submitted_at     : date de soumission
            - total_files      : nombre de fichiers trouvés
            - files_ocr        : liste de résultats OCR par fichier
              (field_title, field_key, url, text_full, information_extraite)
        """
        if not AZURE_VISION_ENDPOINT or not AZURE_VISION_KEY:
            return json.dumps({"error": "Credentials Azure non configurés."}, ensure_ascii=False)

        if not PAPERFORM_API_TOKEN:
            return json.dumps({"error": "PAPERFORM_API_TOKEN non configuré."}, ensure_ascii=False)

        try:
            # ── 1. Récupère la soumission ──────────────────────
            submission_data = _get_paperform_submission(submission_id.strip())
            results         = submission_data.get("results") or {}

            form_id      = results.get("form_id") or ""
            submitted_at = results.get("created_at") or ""

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
                        "field_title":          file_info["field_title"],
                        "field_key":            file_info["field_key"],
                        "field_type":           file_info["field_type"],
                        "url":                  file_info["url"],
                        **ocr_result,
                    })
                except Exception as e:
                    files_ocr.append({
                        "field_title": file_info["field_title"],
                        "field_key":   file_info["field_key"],
                        "url":         file_info["url"],
                        "status":      "error",
                        "error":       str(e),
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
        (ex: toutes les soumissions du jour).

        Args:
            submission_ids_csv : UUIDs de soumissions séparés par des virgules (max 5).
                                 Exemple: "uuid1,uuid2,uuid3"
            language           : Code langue ISO. 'auto' pour détection automatique.

        Returns:
            JSON avec :
            - total    : nombre de soumissions soumises
            - success  : nombre de soumissions traitées avec succès
            - errors   : nombre d'échecs
            - results  : liste de résultats par soumission
        """
        if not AZURE_VISION_ENDPOINT or not AZURE_VISION_KEY:
            return json.dumps({"error": "Credentials Azure non configurés."}, ensure_ascii=False)

        if not PAPERFORM_API_TOKEN:
            return json.dumps({"error": "PAPERFORM_API_TOKEN non configuré."}, ensure_ascii=False)

        # ── Parse des IDs ──────────────────────────────────────
        ids = [sid.strip() for sid in submission_ids_csv.split(",") if sid.strip()]
        if not ids:
            return json.dumps({"error": "Aucun submission_id fourni."}, ensure_ascii=False)

        ids = ids[:5]  # Plafond à 5 — chaque soumission peut avoir plusieurs fichiers

        results       = []
        success_count = 0
        error_count   = 0

        for submission_id in ids:
            try:
                submission_data = _get_paperform_submission(submission_id)
                res_data        = submission_data.get("results") or {}
                files           = _extract_file_urls(submission_data)

                files_ocr = []
                for file_info in files:
                    try:
                        ocr_result = _ocr_file_from_url(file_info["url"], language.strip())
                        files_ocr.append({
                            "field_title": file_info["field_title"],
                            "field_key":   file_info["field_key"],
                            "url":         file_info["url"],
                            **ocr_result,
                        })
                    except Exception as e:
                        files_ocr.append({
                            "field_title": file_info["field_title"],
                            "url":         file_info["url"],
                            "status":      "error",
                            "error":       str(e),
                        })

                results.append({
                    "submission_id": submission_id,
                    "form_id":       res_data.get("form_id") or "",
                    "submitted_at":  res_data.get("created_at") or "",
                    "total_files":   len(files),
                    "status":        "success",
                    "files_ocr":     files_ocr,
                })
                success_count += 1

            except Exception as e:
                results.append({
                    "submission_id": submission_id,
                    "status":        "error",
                    "error":         str(e),
                })
                error_count += 1

        output = {
            "total":   len(ids),
            "success": success_count,
            "errors":  error_count,
            "results": results,
        }
        return json.dumps(output, ensure_ascii=False, indent=2)