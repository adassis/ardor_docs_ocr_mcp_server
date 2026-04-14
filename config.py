# =============================================================
# config.py — Configuration centralisée
# =============================================================
# Unique point d'entrée pour toutes les variables d'environnement.
# Tous les autres fichiers importent depuis ici.
# Avantage : si une variable change de nom, on ne modifie qu'ici.
# =============================================================

import os

# ── Serveur MCP ───────────────────────────────────────────────
PORT = int(os.environ.get("PORT", 8000))
# Port d'écoute du serveur.
# Railway injecte automatiquement sa propre valeur via cette variable.
# Valeur par défaut : 8000 pour les tests en local.

MCP_BEARER_TOKEN = os.environ.get("MCP_BEARER_TOKEN", "")
# Token secret pour sécuriser l'accès au serveur MCP.
# Dust l'envoie dans l'en-tête : Authorization: Bearer <token>
# Si vide : authentification désactivée (à éviter en production).

# ── Azure Computer Vision ─────────────────────────────────────
AZURE_VISION_ENDPOINT = os.environ.get("AZURE_VISION_ENDPOINT", "").rstrip("/")
# URL de la ressource Azure Computer Vision.
# Ex: https://mon-ocr.cognitiveservices.azure.com
# .rstrip("/") évite les doubles // dans les URLs construites dynamiquement.

AZURE_VISION_KEY = os.environ.get("AZURE_VISION_KEY", "")
# Clé d'authentification Azure (Clé 1 ou Clé 2 depuis le portail Azure).