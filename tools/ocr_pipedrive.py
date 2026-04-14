# =============================================================
# tools/ocr_pipedrive.py — Outil MCP : OCR pièce jointe Pipedrive
# =============================================================
# Expose un seul outil MCP : read_pipedrive_attachment_ocr
# Dépend de : config, utils/azure, utils/text
# =============================================================

import json
import mimetypes
import re
import requests

from config import (
    AZURE_VISION_ENDPOINT,
    AZURE_VISION_KEY,
    PIPEDRIVE_API_TOKEN,
    PIPEDRIVE_SUBDOMAIN,
)
from utils.azure import azure_read_analyze, extract_text_from_azure_result
from utils.text import extract_kv_from_lines, extract_tokens


def _download_pipedrive_attachment(file_id: str):
    """
    Télécharge une pièce jointe depuis Pipedrive via l'API v1.

    L'URL suit le pattern :
    https://{subdomain}.pipedrive.com/api/v1/mailbox/mailAttachments/{id}/download

    Le token API est passé en query string (méthode standard Pipedrive v1).
    allow_redirects=True car Pipedrive redirige souvent vers un CDN externe.

    Returns:
        tuple (content, content_type, extension)

    Raises:
        RuntimeError : token manquant, erreur HTTP, ou réponse HTML inattendue
    """
    if not PIPEDRIVE_API_TOKEN:
        raise RuntimeError(
            "PIPEDRIVE_API_TOKEN non configuré. "
            "Ajoutez cette variable d'environnement sur Railway."
        )
    if not PIPEDRIVE_SUBDOMAIN:
        raise RuntimeError(
            "PIPEDRIVE_SUBDOMAIN non configuré. "
            "Ajoutez cette variable d'environnement sur Railway (ex: 'seraphin')."
        )

    url = (
        f"https://{PIPEDRIVE_SUBDOMAIN}.pipedrive.com"
        f"/api/v1/mailbox/mailAttachments/{file_id}/download"
    )

    # Le token est passé en query string : ?api_token=xxx
    params = {"api_token": PIPEDRIVE_API_TOKEN}
    r = requests.get(url, params=params, timeout=90, allow_redirects=True)

    if r.status_code != 200:
        raise RuntimeError(
            f"Echec téléchargement Pipedrive (HTTP {r.status_code}): {r.text[:400]}"
        )

    content = r.content
    ct = r.headers.get("Content-Type", "") or ""

    # Pipedrive renvoie du HTML si le token est invalide ou expiré
    # → erreur explicite plutôt qu'un OCR vide
    if ct.startswith("text/html") or b"<html" in content[:512].lower():
        raise RuntimeError(
            f"Pipedrive a renvoyé du HTML au lieu du fichier. "
            f"Vérifiez PIPEDRIVE_API_TOKEN. "
            f"Content-Type={ct}. Extrait: {content[:200]!r}"
        )

    # Détection de l'extension depuis le Content-Type
    ext = mimetypes.guess_extension(ct.split(";")[0].strip()) or ""
    if not ext:
        # Fallback : extraire l'extension depuis l'URL
        m = re.search(r"\.([a-zA-Z0-9]{2,5})(?:\?|$)", url)
        if m:
            ext = "." + m.group(1).lower()

    return content, ct, ext


def register(mcp):
    """
    Enregistre l'outil read_pipedrive_attachment_ocr sur l'instance FastMCP.

    Appelé au démarrage depuis server.py :
        import tools.ocr_pipedrive
        tools.ocr_pipedrive.register(mcp)
    """

    @mcp.tool()
    def read_pipedrive_attachment_ocr(file_id: str, language: str = "auto") -> str:
        """
        Lit une pièce jointe Pipedrive via son identifiant et en extrait
        le texte par OCR Azure Computer Vision.

        Utilisez cet outil quand l'utilisateur mentionne un identifiant
        de pièce jointe Pipedrive et souhaite en connaître le contenu.

        Args:
            file_id  : Identifiant numérique de la pièce jointe Pipedrive.
                       Exemple : "123456789"
            language : Code langue ISO ('fr', 'en', 'nl'...).
                       'auto' pour détection automatique (défaut).

        Returns:
            JSON string avec texte extrait et paires clé-valeur.
        """
        # ── Vérification des credentials ──────────────────────
        if not AZURE_VISION_ENDPOINT or not AZURE_VISION_KEY:
            return json.dumps({
                "error": (
                    "Credentials Azure non configurés. "
                    "Vérifiez AZURE_VISION_ENDPOINT et AZURE_VISION_KEY sur Railway."
                )
            }, ensure_ascii=False)

        if not PIPEDRIVE_API_TOKEN or not PIPEDRIVE_SUBDOMAIN:
            return json.dumps({
                "error": (
                    "Credentials Pipedrive non configurés. "
                    "Vérifiez PIPEDRIVE_API_TOKEN et PIPEDRIVE_SUBDOMAIN sur Railway."
                )
            }, ensure_ascii=False)

        try:
            # ── 1. Téléchargement depuis Pipedrive ─────────────
            blob, content_type, ext = _download_pipedrive_attachment(file_id.strip())

            # ── 2. OCR Azure ───────────────────────────────────
            # Réutilise azure_read_analyze de utils/azure.py
            result = azure_read_analyze(
                blob, AZURE_VISION_ENDPOINT, AZURE_VISION_KEY, language.strip()
            )

            # ── 3. Extraction du texte ─────────────────────────
            # Réutilise extract_text_from_azure_result de utils/azure.py
            text_full, pages_struct = extract_text_from_azure_result(result)

            # ── 4. Construction de la liste des lignes ─────────
            lines = []
            if pages_struct:
                for pg in pages_struct:
                    lines.extend(pg["lines"])
            else:
                lines = [
                    l.strip()
                    for l in (text_full or "").splitlines()
                    if l and l.strip()
                ]

            # ── 5. Extraction clé-valeur et tokens ─────────────
            # Réutilise les fonctions factorisées dans utils/text.py
            kv     = extract_kv_from_lines(lines)
            tokens = extract_tokens(lines, max_tokens=2000)

            # ── 6. Construction de la réponse JSON ─────────────
            output = {
                "file_id": file_id,
                "meta": {
                    "content_type":      content_type,
                    "guessed_extension": ext,
                    "file_size_bytes":   len(blob),
                    "char_count":        len(text_full),
                    "line_count":        len(lines),
                    "token_count":       len(tokens),
                    "azure_status":      result.get("status"),
                    "pipedrive_subdomain": PIPEDRIVE_SUBDOMAIN,
                },
                "text_full":            text_full,
                "pages":                pages_struct,
                "lines":                lines,
                "information_extraite": kv,
                "tokens":               tokens,
            }

            return json.dumps(output, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps(
                {"error": str(e), "file_id": file_id},
                ensure_ascii=False
            )
    @mcp.tool()
    def read_pipedrive_attachments_ocr_bulk(file_ids_csv: str, language: str = "auto") -> str:
        """
        Effectue l'OCR sur plusieurs pièces jointes Pipedrive en un seul appel.
        Conçu pour traiter la liste retournée par list_deal_attachments
        (outil du serveur MCP Pipedrive) sans appels individuels répétés.

        Args:
            file_ids_csv : IDs de pièces jointes séparés par des virgules (max 10).
                        Exemple: "123456,789012,345678"
                        Les IDs viennent du champ "attachment_ids" de list_deal_attachments.
            language     : Code langue ISO ('fr', 'en', 'nl'...) appliqué à tous.
                        'auto' pour détection automatique (défaut).

        Returns:
            JSON avec :
            - total   : nombre de fichiers soumis
            - success : nombre d'OCR réussis
            - errors  : nombre d'échecs
            - results : liste de {file_id, status, text_full, information_extraite, error}
        """
        if not AZURE_VISION_ENDPOINT or not AZURE_VISION_KEY:
            return json.dumps({"error": "Credentials Azure non configurés."}, ensure_ascii=False)

        if not PIPEDRIVE_API_TOKEN or not PIPEDRIVE_SUBDOMAIN:
            return json.dumps({"error": "Credentials Pipedrive non configurés."}, ensure_ascii=False)

        # ── Parse et validation des IDs ───────────────────────────
        file_ids = [fid.strip() for fid in file_ids_csv.split(",") if fid.strip()]
        if not file_ids:
            return json.dumps({"error": "Aucun file_id fourni."}, ensure_ascii=False)

        file_ids = file_ids[:10]  # Plafond à 10 pour éviter les timeouts

        results = []
        success_count = 0
        error_count   = 0

        for file_id in file_ids:
            try:
                # ── Téléchargement depuis Pipedrive ────────────────
                blob, content_type, ext = _download_pipedrive_attachment(file_id)

                # ── OCR Azure ──────────────────────────────────────
                result     = azure_read_analyze(blob, AZURE_VISION_ENDPOINT, AZURE_VISION_KEY, language.strip())
                text_full, pages_struct = extract_text_from_azure_result(result)

                lines = []
                if pages_struct:
                    for pg in pages_struct:
                        lines.extend(pg["lines"])
                else:
                    lines = [l.strip() for l in (text_full or "").splitlines() if l and l.strip()]

                kv = extract_kv_from_lines(lines)

                results.append({
                    "file_id":             file_id,
                    "status":              "success",
                    "content_type":        content_type,
                    "char_count":          len(text_full),
                    "text_full":           text_full,
                    "information_extraite": kv,
                    "error":               None,
                })
                success_count += 1

            except Exception as e:
                results.append({
                    "file_id": file_id,
                    "status":  "error",
                    "error":   str(e),
                })
                error_count += 1

        output = {
            "total":   len(file_ids),
            "success": success_count,
            "errors":  error_count,
            "results": results,
        }
        return json.dumps(output, ensure_ascii=False, indent=2)