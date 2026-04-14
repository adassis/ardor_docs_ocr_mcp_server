# =============================================================
# server.py — Point d'entrée du serveur MCP
# =============================================================
# Ce fichier a une seule responsabilité : orchestrer.
# Il ne contient AUCUNE logique métier.
#
# Pour ajouter un nouvel outil :
#   1. Créer tools/mon_outil.py avec une fonction register(mcp)
#   2. Ajouter les 2 lignes import + register() ci-dessous
#   C'est tout.
# =============================================================

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from config import PORT, MCP_BEARER_TOKEN

# ── Import des modules d'outils ───────────────────────────────
import tools.ocr_url
import tools.ocr_pipedrive



# ── Initialisation du serveur MCP ─────────────────────────────
mcp = FastMCP(
    name="azure-ocr-server",
    host="0.0.0.0",
    port=PORT,
    instructions=(
        "Serveur OCR Azure Computer Vision. "
        "Utilisez read_document_ocr pour extraire le texte "
        "d'un document image ou PDF depuis une URL publique."
    )
)


# ── Enregistrement des outils ─────────────────────────────────
tools.ocr_url.register(mcp)
tools.ocr_pipedrive.register(mcp)


# ── Middleware d'authentification Bearer Token ────────────────
class BearerAuthMiddleware(BaseHTTPMiddleware):
    """
    Intercepte toutes les requêtes HTTP entrantes.
    Vérifie la présence et la validité du Bearer Token
    avant de les transmettre au serveur MCP.
    """
    async def dispatch(self, request, call_next):
        if MCP_BEARER_TOKEN:
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[7:].strip() != MCP_BEARER_TOKEN:
                return JSONResponse({"error": "Non autorisé"}, status_code=401)
        return await call_next(request)


# ── Démarrage ─────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"🚀 Serveur MCP démarré sur le port {PORT}")
    print(f"🔐 Auth Bearer Token : {'Activée' if MCP_BEARER_TOKEN else 'DÉSACTIVÉE ⚠️'}")

    app = mcp.streamable_http_app()
    app.add_middleware(BearerAuthMiddleware)
    uvicorn.run(app, host="0.0.0.0", port=PORT)