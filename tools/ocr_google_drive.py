import json
import re
import requests

from config import (
    AZURE_VISION_ENDPOINT,
    AZURE_VISION_KEY,
    GOOGLE_SERVICE_ACCOUNT_JSON,
)
from utils.azure import azure_read_analyze, extract_text_from_azure_result
from utils.text import extract_kv_from_lines, extract_tokens
from utils.quality import (
    extract_ocr_metrics,
    build_image_quality_estimate,
    build_document_quality_estimate,
    compute_global_document_score,
    build_processing_diagnostics,
)

SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/tiff",
    "image/bmp",
    "image/gif",
    "image/webp",
}


# =============================================================
# EXTRACTION D'ID
# =============================================================

def _extract_gdrive_file_id(url: str) -> str:
    m = re.search(r'/d/([a-zA-Z0-9_-]{10,})', url)
    if m:
        return m.group(1)
    m = re.search(r'[?&]id=([a-zA-Z0-9_-]{10,})', url)
    if m:
        return m.group(1)
    raise ValueError(
        f"Impossible d'extraire l'ID du fichier depuis : {url}\n"
        "Formats supportés :\n"
        "  • https://drive.google.com/file/d/FILE_ID/view?usp=sharing\n"
        "  • https://drive.google.com/open?id=FILE_ID"
    )


def _extract_gdrive_folder_id(url: str) -> str:
    if re.match(r'^[a-zA-Z0-9_-]{10,}$', url.strip()):
        return url.strip()
    m = re.search(r'/folders/([a-zA-Z0-9_-]{10,})', url)
    if m:
        return m.group(1)
    m = re.search(r'/d/([a-zA-Z0-9_-]{10,})', url)
    if m:
        return m.group(1)
    m = re.search(r'[?&]id=([a-zA-Z0-9_-]{10,})', url)
    if m:
        return m.group(1)
    raise ValueError(
        f"Impossible d'extraire l'ID du dossier depuis : {url}\n"
        "Formats supportés :\n"
        "  • https://drive.google.com/drive/folders/FOLDER_ID\n"
        "  • https://drive.google.com/drive/folders/FOLDER_ID?usp=sharing"
    )


# =============================================================
# AUTHENTIFICATION
# =============================================================

def _get_google_access_token() -> str:
    try:
        import google.oauth2.service_account
        import google.auth.transport.requests
    except ImportError:
        raise RuntimeError(
            "La librairie 'google-auth' n'est pas installée. "
            "Ajoutez 'google-auth>=2.0.0' dans requirements.txt."
        )
    try:
        service_account_info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"GOOGLE_SERVICE_ACCOUNT_JSON invalide (JSON malformé) : {e}\n"
            "Copiez le contenu COMPLET du fichier JSON du Service Account."
        )
    credentials = google.oauth2.service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token


# =============================================================
# LISTING DU DOSSIER
# =============================================================

def _list_files_in_folder(folder_id: str, access_token: str) -> list:
    files = []
    page_token = None
    headers = {"Authorization": f"Bearer {access_token}"}

    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "fields": "nextPageToken, files(id, name, mimeType, size)",
            "pageSize": 1000,
            "supportsAllDrives": True,         # ← AJOUTÉ : accès aux Shared Drives
            "includeItemsFromAllDrives": True,  # ← AJOUTÉ : inclut les fichiers des Shared Drives
        }
        if page_token:
            params["pageToken"] = page_token

        r = requests.get(
            "https://www.googleapis.com/drive/v3/files",
            headers=headers,
            params=params,
            timeout=30,
        )

        if r.status_code == 403:
            raise RuntimeError(
                "Accès refusé au dossier Google Drive (403). "
                "Le Service Account n'a pas accès à ce dossier.\n"
                "→ Dans Google Drive : clic droit sur le dossier > Partager\n"
                "→ Ajoutez l'email du Service Account comme 'Lecteur'."
            )
        if not r.ok:
            raise RuntimeError(
                f"Erreur API Google Drive HTTP {r.status_code} : {r.text[:300]}"
            )

        data = r.json()
        files.extend(data.get("files", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return files


def _list_supported_files_in_folder(folder_id: str, access_token: str) -> tuple:
    all_files = _list_files_in_folder(folder_id, access_token)
    supported = []
    skipped = []

    for f in all_files:
        mime = f.get("mimeType", "")
        if mime in SUPPORTED_MIME_TYPES:
            supported.append(f)
        else:
            skipped.append({
                "id": f.get("id"),
                "name": f.get("name"),
                "mimeType": mime,
                "reason": (
                    "Fichier Google natif (Docs/Sheets/Slides). "
                    "Exportez en PDF : Fichier > Télécharger > PDF."
                    if mime.startswith("application/vnd.google-apps")
                    else (
                        "Type MIME non supporté par Azure OCR. "
                        "Supportés : PDF, JPG, PNG, TIFF, BMP, GIF, WebP."
                    )
                ),
            })

    return supported, skipped


# =============================================================
# TÉLÉCHARGEMENT
# =============================================================

def _download_gdrive_file_via_api(file_id: str, access_token: str) -> tuple:
    headers = {"Authorization": f"Bearer {access_token}"}
    r = requests.get(
        f"https://www.googleapis.com/drive/v3/files/{file_id}",
        headers=headers,
        params={
            "alt": "media",
            "supportsAllDrives": True,  # ← AJOUTÉ : accès aux fichiers des Shared Drives
        },
        timeout=120,
        allow_redirects=True,
    )
    if r.status_code == 404:
        raise RuntimeError(f"Fichier introuvable via l'API Drive (404). ID : {file_id}")
    if r.status_code == 403:
        raise RuntimeError(f"Accès refusé au fichier (403). ID : {file_id}")
    if not r.ok:
        raise RuntimeError(f"Erreur HTTP {r.status_code} pour le fichier {file_id}.")
    content_type = r.headers.get("Content-Type", "application/octet-stream") or ""
    return r.content, content_type


def _download_gdrive_file_public(file_id: str) -> tuple:
    session = requests.Session()
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    r = session.get(download_url, headers=headers, timeout=90, allow_redirects=True)

    if r.status_code == 404:
        raise RuntimeError("Fichier Google Drive introuvable (404).")
    if r.status_code == 403:
        raise RuntimeError(
            "Accès refusé (403). Configurez GOOGLE_SERVICE_ACCOUNT_JSON "
            "pour accéder aux fichiers privés."
        )
    if r.status_code != 200:
        raise RuntimeError(f"Erreur HTTP {r.status_code} Google Drive.")

    content_type = r.headers.get("Content-Type", "") or ""
    is_html = (
        content_type.startswith("text/html")
        or b"<html" in r.content[:512].lower()
    )

    if is_html:
        confirm_token = None
        for cookie in session.cookies:
            if cookie.name == "download_warning":
                confirm_token = cookie.value
                break
        if not confirm_token:
            m = re.search(r'confirm=([0-9A-Za-z_\-]+)', r.text)
            if m:
                confirm_token = m.group(1)
        if not confirm_token:
            m = re.search(r'"downloadUrl":"(https://[^"]+)"', r.text)
            if m:
                direct_url = m.group(1).replace('\\u003d', '=').replace('\\u0026', '&')
                r2 = session.get(direct_url, headers=headers, timeout=120, allow_redirects=True)
                ct2 = r2.headers.get("Content-Type", "") or ""
                if r2.ok and "text/html" not in ct2:
                    return r2.content, ct2
        if confirm_token:
            r = session.get(
                f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}",
                headers=headers, timeout=120, allow_redirects=True,
            )
            content_type = r.headers.get("Content-Type", "") or ""
            if "text/html" in content_type or b"<html" in r.content[:512].lower():
                raise RuntimeError(
                    "Google Drive a renvoyé du HTML après confirmation. "
                    "Configurez GOOGLE_SERVICE_ACCOUNT_JSON pour les fichiers privés."
                )
        else:
            raise RuntimeError(
                "Google Drive a renvoyé une page HTML. Le fichier est peut-être privé.\n"
                "→ Configurez GOOGLE_SERVICE_ACCOUNT_JSON pour les fichiers privés.\n"
                "→ Ou vérifiez le partage 'Tout le monde avec le lien'."
            )

    if not r.content:
        raise RuntimeError("Le fichier téléchargé est vide.")

    return r.content, content_type


def _download_gdrive_file(file_id: str) -> tuple:
    if GOOGLE_SERVICE_ACCOUNT_JSON:
        token = _get_google_access_token()
        return _download_gdrive_file_via_api(file_id, token)
    else:
        return _download_gdrive_file_public(file_id)


# =============================================================
# OCR D'UN FICHIER (fonction partagée)
# =============================================================

def _ocr_one_file(blob: bytes, content_type: str, language: str) -> dict:
    result = azure_read_analyze(
        blob, AZURE_VISION_ENDPOINT, AZURE_VISION_KEY, language.strip()
    )
    text_full, pages_struct = extract_text_from_azure_result(result)

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

    kv = extract_kv_from_lines(lines)
    tokens = extract_tokens(lines, max_tokens=2000)
    ocr_metrics = extract_ocr_metrics(result)

    return {
        "meta": {
            "content_type": content_type,
            "file_size_bytes": len(blob),
            "char_count": len(text_full),
            "line_count": len(lines),
            "token_count": len(tokens),
            "azure_status": result.get("status"),
        },
        "text_full": text_full,
        "pages": pages_struct,
        "lines": lines,
        "information_extraite": kv,
        "tokens": tokens,
        "qualite_image": build_image_quality_estimate(blob, result, text_full, ocr_metrics),
        "qualite_document": build_document_quality_estimate(text_full, lines, kv, ocr_metrics),
        "score_global": compute_global_document_score(ocr_metrics, text_full, lines),
        "diagnostics": build_processing_diagnostics(result, text_full, lines, tokens, ocr_metrics),
    }


# =============================================================
# ENREGISTREMENT DES OUTILS MCP
# =============================================================

def register(mcp):

    # ──────────────────────────────────────────────────────────
    # OUTIL 1 : Fichier unique
    # ──────────────────────────────────────────────────────────

    @mcp.tool()
    def read_google_drive_ocr(file_url: str, language: str = "auto") -> str:
        """
        Lit un fichier Google Drive et en extrait le texte par OCR
        Azure Computer Vision.

        Supporte les fichiers privés si GOOGLE_SERVICE_ACCOUNT_JSON
        est configuré et que le Service Account a accès au fichier.
        Sans Service Account, le fichier doit être partagé publiquement.

        Types supportés : PDF, JPG, PNG, TIFF, BMP, GIF, WebP.
        Google Docs/Sheets/Slides natifs non supportés
        (exportez en PDF d'abord via Fichier > Télécharger > PDF).

        Args:
            file_url : URL de partage Google Drive.
                       • https://drive.google.com/file/d/FILE_ID/view?usp=sharing
                       • https://drive.google.com/open?id=FILE_ID
            language : Code langue ISO ('fr', 'en', 'nl'...) ou 'auto'.

        Returns:
            JSON avec texte extrait, paires clé-valeur et scores qualité.
        """
        if not AZURE_VISION_ENDPOINT or not AZURE_VISION_KEY:
            return json.dumps({"error": "Credentials Azure non configurés."}, ensure_ascii=False)

        try:
            file_id = _extract_gdrive_file_id(file_url.strip())
            blob, content_type = _download_gdrive_file(file_id)
            ocr_result = _ocr_one_file(blob, content_type, language)

            output = {
                "file_url": file_url,
                "gdrive_file_id": file_id,
                "auth_mode": "service_account" if GOOGLE_SERVICE_ACCOUNT_JSON else "public",
                **ocr_result,
            }
            return json.dumps(output, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e), "file_url": file_url}, ensure_ascii=False)


    # ──────────────────────────────────────────────────────────
    # OUTIL 2 : Tous les fichiers d'un dossier (avec pagination)
    # ──────────────────────────────────────────────────────────

    @mcp.tool()
    def read_google_drive_folder_ocr(
        folder_url: str,
        language: str = "auto",
        max_files: int = 5,
        offset: int = 0,
    ) -> str:
        """
        Liste et OCRise les fichiers d'un dossier Google Drive par batch.

        Conçu pour traiter les gros dossiers sans timeout Railway :
        chaque appel traite un petit nombre de fichiers (max_files),
        à partir d'une position donnée (offset).

        L'agent doit enchaîner les appels tant que has_more = true :
          Appel 1 : offset=0,  max_files=5
          Appel 2 : offset=5,  max_files=5   (si has_more=true)
          Appel 3 : offset=10, max_files=5   (si has_more=true)
          ... jusqu'à has_more=false

        Nécessite GOOGLE_SERVICE_ACCOUNT_JSON configuré sur Railway.
        Le Service Account doit avoir accès au dossier :
        Dans Google Drive → clic droit sur le dossier → Partager →
        ajouter l'email du Service Account comme Lecteur.

        Fichiers ignorés automatiquement : Google Docs/Sheets/Slides
        natifs et types non supportés par Azure OCR.

        Args:
            folder_url : URL du dossier Google Drive.
                         • https://drive.google.com/drive/folders/FOLDER_ID
                         • https://drive.google.com/drive/folders/FOLDER_ID?usp=sharing
                         • L'ID du dossier directement
            language   : Code langue ISO ('fr', 'en', 'nl'...) ou 'auto'.
            max_files  : Nombre de fichiers à traiter par appel (défaut: 5).
                         Gardez une valeur basse (5-10) pour éviter les timeouts.
            offset     : Position de départ dans la liste (défaut: 0).
                         Utilisez next_offset de la réponse précédente.

        Returns:
            JSON avec :
            - has_more                : true si d'autres fichiers restent
            - next_offset             : valeur d'offset pour l'appel suivant
            - summary.total_supported : total fichiers OCRisables dans le dossier
            - summary.processed       : fichiers traités dans CE batch
            - summary.offset          : position de départ de ce batch
            - files_skipped           : fichiers ignorés (premier appel uniquement)
            - results                 : OCR des fichiers de ce batch
            - errors                  : fichiers en erreur dans ce batch
        """
        if not AZURE_VISION_ENDPOINT or not AZURE_VISION_KEY:
            return json.dumps({"error": "Credentials Azure non configurés."}, ensure_ascii=False)

        if not GOOGLE_SERVICE_ACCOUNT_JSON:
            return json.dumps({
                "error": (
                    "GOOGLE_SERVICE_ACCOUNT_JSON non configuré. "
                    "Le listing de dossier nécessite un Service Account Google."
                )
            }, ensure_ascii=False)

        try:
            folder_id = _extract_gdrive_folder_id(folder_url.strip())
            access_token = _get_google_access_token()

            supported_files, skipped_files = _list_supported_files_in_folder(
                folder_id, access_token
            )

            total_supported = len(supported_files)
            batch = supported_files[offset: offset + max_files]
            next_offset = offset + len(batch)
            has_more = next_offset < total_supported

            results = []
            errors = []

            for idx, file_info in enumerate(batch):
                file_id = file_info["id"]
                file_name = file_info.get("name", "inconnu")
                file_size = int(file_info.get("size", 0) or 0)

                try:
                    blob, content_type = _download_gdrive_file_via_api(
                        file_id, access_token
                    )
                    ocr_result = _ocr_one_file(blob, content_type, language)
                    results.append({
                        "index": offset + idx + 1,
                        "file_id": file_id,
                        "file_name": file_name,
                        "file_size_bytes": file_size,
                        **ocr_result,
                    })
                except Exception as e:
                    errors.append({
                        "file_id": file_id,
                        "file_name": file_name,
                        "error": str(e),
                    })

            output = {
                "folder_url": folder_url,
                "folder_id": folder_id,
                "auth_mode": "service_account",
                "has_more": has_more,
                "next_offset": next_offset if has_more else None,
                "summary": {
                    "total_supported": total_supported,
                    "total_skipped": len(skipped_files),
                    "offset": offset,
                    "batch_size": len(batch),
                    "processed": len(results),
                    "errors": len(errors),
                },
                "files_skipped": skipped_files if offset == 0 else [],
                "results": results,
                "errors": errors,
            }

            return json.dumps(output, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps(
                {"error": str(e), "folder_url": folder_url},
                ensure_ascii=False
            )