import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from config import PORT, MCP_BEARER_TOKEN

import tools.ocr_ardor_docs
import tools.ocr_paperform_attachments
import tools.ocr_pipedrive_attachments
import tools.ocr_google_drive

mcp = FastMCP(
    name="yago_ocr_read_documents",
    host="0.0.0.0",
    port=PORT,
    instructions=(
        "Serveur OCR Azure Computer Vision. "
        "Utilisez read_document_ocr pour extraire le texte "
        "d'un document image ou PDF depuis une URL publique."
    )
)

tools.ocr_ardor_docs.register(mcp)
tools.ocr_paperform_attachments.register(mcp)
tools.ocr_pipedrive_attachments.register(mcp)
tools.ocr_google_drive.register(mcp)

class BearerAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if MCP_BEARER_TOKEN:
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[7:].strip() != MCP_BEARER_TOKEN:
                return JSONResponse({"error": "Non autorisé"}, status_code=401)
        return await call_next(request)

if __name__ == "__main__":
    print(f"🚀 Serveur MCP démarré sur le port {PORT}")
    print(f"🔐 Auth Bearer Token : {'Activée' if MCP_BEARER_TOKEN else 'DÉSACTIVÉE ⚠️'}")
    app = mcp.streamable_http_app()
    app.add_middleware(BearerAuthMiddleware)
    uvicorn.run(app, host="0.0.0.0", port=PORT)