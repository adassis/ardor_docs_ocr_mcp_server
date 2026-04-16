# =============================================================
# tools/ocr_url.py — Outil MCP : OCR depuis URL publique
# =============================================================
# Expose un seul outil MCP : read_document_ocr
# Dépend de : config, utils/azure, utils/text, utils/quality
# =============================================================

import json
import mimetypes
import re
import requests

from config import AZURE_VISION_ENDPOINT, AZURE_VISION_KEY
from utils.azure import azure_read_analyze, extract_text_from_azure_result
from utils.text import extract_kv_from_lines, extract_tokens
from utils.quality import (
    extract_ocr_metrics,
    build_image_quality_estimate,
    build_document_quality_estimate,
    compute_global_document_score,
    build_processing_diagnostics,
)


def _download_file_from_url(url: str):
    """
    Télécharge un fichier depuis une URL publique.

    Privée à ce module (préfixe _) : utilisée uniquement par read_document_ocr.
    La version Pipedrive est dans tools/ocr_pipedrive.py.

    Returns:
        tuple (content, content_type, extension)

    Raises:
        RuntimeError : si le téléchargement échoue ou renvoie du HTML
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=90, allow_redirects=True)

    if r.status_code != 200:
        raise RuntimeError(
            f"Echec téléchargement URL (HTTP {r.status_code}): {r.text[:400]}"
        )

    content = r.content
    ct = r.headers.get("Content-Type", "") or ""

    # Certaines URLs renvoient du HTML (page d'erreur, redirection login...)
    # au lieu du fichier attendu → erreur explicite
    if ct.startswith("text/html") or b"<html" in content[:512].lower():
        raise RuntimeError(
            f"L'URL a renvoyé du HTML. Content-Type={ct}. "
            f"Extrait: {content[:200]!r}"
        )

    # Détection de l'extension depuis le Content-Type
    ext = mimetypes.guess_extension(ct.split(";")[0].strip()) or ""
    if not ext:
        # Fallback : extraire l'extension depuis l'URL elle-même
        m = re.search(r"\.([a-zA-Z0-9]{2,5})(?:\?|$)", url)
        if m:
            ext = "." + m.group(1).lower()

    return content, ct, ext


def register(mcp):
    """
    Enregistre l'outil read_document_ocr sur l'instance FastMCP.

    Appelé une seule fois au démarrage depuis server.py :
        import tools.ocr_url
        tools.ocr_url.register(mcp)

    Args:
        mcp : instance FastMCP initialisée dans server.py
    """

    @mcp.tool()
    def read_document_ocr(file_url: str, language: str = "auto") -> str:
        """
        Lit un document depuis une URL publique via Azure Computer Vision OCR.

        Utilisez cet outil quand l'utilisateur fournit une URL directe
        vers une image ou un PDF accessible publiquement.

        Args:
            file_url : URL publique du fichier (JPG, PNG, PDF, TIFF...).
            language : Code langue ISO ('fr', 'en', 'nl'...).
                       'auto' pour détection automatique (défaut).

        Returns:
            JSON string avec texte extrait, paires clé-valeur et scores qualité.
        """
        # ── Vérification des credentials Azure ────────────────
        if not AZURE_VISION_ENDPOINT or not AZURE_VISION_KEY:
            return json.dumps({
                "error": (
                    "Credentials Azure non configurés. "
                    "Vérifiez AZURE_VISION_ENDPOINT et AZURE_VISION_KEY sur Railway."
                )
            }, ensure_ascii=False)

        try:
            # ── 1. Téléchargement du fichier ───────────────────
            blob, content_type, ext = _download_file_from_url(file_url.strip())

            # ── 2. OCR Azure ───────────────────────────────────
            result = azure_read_analyze(
                blob, AZURE_VISION_ENDPOINT, AZURE_VISION_KEY, language.strip()
            )

            # ── 3. Extraction du texte ─────────────────────────
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
            # Fonctions factorisées dans utils/text.py,
            # réutilisables par tous les outils OCR du projet.
            kv     = extract_kv_from_lines(lines)
            tokens = extract_tokens(lines, max_tokens=2000)

            # ── 6. Calcul des scores qualité ───────────────────
            # Fonctions factorisées dans utils/quality.py
            ocr_metrics               = extract_ocr_metrics(result)
            image_quality_estimate    = build_image_quality_estimate(blob, result, text_full, ocr_metrics)
            document_quality_estimate = build_document_quality_estimate(text_full, lines, kv, ocr_metrics)
            global_document_score     = compute_global_document_score(ocr_metrics, text_full, lines)
            processing_diagnostics    = build_processing_diagnostics(result, text_full, lines, tokens, ocr_metrics)

            # ── 7. Construction de la réponse JSON ─────────────
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

        except Exception as e:
            return json.dumps(
                {"error": str(e), "file_url": file_url},
                ensure_ascii=False
            )
    @mcp.tool()
    def read_documents_ocr_bulk(urls_csv: str, language: str = "auto") -> str:
        """
        Effectue l'OCR sur plusieurs documents publics en un seul appel.
        Conçu pour éviter les appels individuels répétés quand on a
        plusieurs URLs à analyser.

        Args:
            urls_csv : URLs séparées par des virgules (max 10).
                    Exemple: "https://site.com/doc1.pdf,https://site.com/doc2.pdf"
            language : Code langue ISO ('fr', 'en', 'nl'...) appliqué à tous.
                    'auto' pour détection automatique (défaut).

        Returns:
            JSON avec :
            - total        : nombre d'URLs soumises
            - success      : nombre d'OCR réussis
            - errors       : nombre d'échecs
            - results      : liste de {url, status, text_full, information_extraite, error}
        """
        if not AZURE_VISION_ENDPOINT or not AZURE_VISION_KEY:
            return json.dumps({"error": "Credentials Azure non configurés."}, ensure_ascii=False)

        # ── Parse et validation des URLs ──────────────────────────
        urls = [u.strip() for u in urls_csv.split(",") if u.strip()]
        if not urls:
            return json.dumps({"error": "Aucune URL fournie."}, ensure_ascii=False)

        urls = urls[:10]  # Plafond à 10 pour éviter les timeouts

        results = []
        success_count = 0
        error_count   = 0

        for url in urls:
            try:
                # ── Téléchargement ─────────────────────────────────
                r = requests.get(url, timeout=60, allow_redirects=True)
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}")

                blob = r.content

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
                    "url":                 url,
                    "status":              "success",
                    "char_count":          len(text_full),
                    "text_full":           text_full,
                    "information_extraite": kv,
                    "error":               None,
                })
                success_count += 1

            except Exception as e:
                results.append({
                    "url":    url,
                    "status": "error",
                    "error":  str(e),
                })
                error_count += 1

        output = {
            "total":   len(urls),
            "success": success_count,
            "errors":  error_count,
            "results": results,
        }
        return json.dumps(output, ensure_ascii=False, indent=2)