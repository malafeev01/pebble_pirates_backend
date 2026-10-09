"""
Beach Treasure Finder — Gemini proxy backend.

Accepts a photo from the iOS app, checks the caller's daily free-tier
limit, asks Gemini to identify the rock/shell, and returns a small
structured JSON result the app can render directly.

The API key never reaches the iOS app — it lives only here, as an
environment variable.
"""

import io
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import date
from typing import Optional

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuration (all via environment variables — see .env.example)
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
APP_SHARED_SECRET = os.environ.get("APP_SHARED_SECRET")  # optional but recommended
DAILY_FREE_LIMIT = int(os.environ.get("DAILY_FREE_LIMIT", "5"))
DB_PATH = os.environ.get("DB_PATH", "usage.db")
MAX_IMAGE_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "1024"))

client = genai.Client(api_key=GEMINI_API_KEY)
app = FastAPI(title="Beach Treasure Finder — Gemini proxy")


# ---------------------------------------------------------------------------
# Usage tracking — one row per device per day, plain SQLite (no extra infra)
# ---------------------------------------------------------------------------

@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usage (
                device_id TEXT NOT NULL,
                day       TEXT NOT NULL,
                count     INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (device_id, day)
            )
            """
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_and_increment_usage(device_id: str) -> int:
    """Increments and returns today's count for this device."""
    today = date.today().isoformat()
    with db() as conn:
        conn.execute(
            """
            INSERT INTO usage (device_id, day, count) VALUES (?, ?, 1)
            ON CONFLICT(device_id, day) DO UPDATE SET count = count + 1
            """,
            (device_id, today),
        )
        count = conn.execute(
            "SELECT count FROM usage WHERE device_id = ? AND day = ?",
            (device_id, today),
        ).fetchone()[0]
    return count


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------

PROMPT = """Ты помогаешь детскому приложению определять камни, ракушки и другие находки с пляжа по фотографии.

Ответь СТРОГО в формате JSON, без markdown-разметки, без пояснений вне JSON:
{
  "name": "короткое название находки по-русски, например 'Кварцевая галька'",
  "category": "stone" | "shell" | "other",
  "confidence": число от 0.0 до 1.0,
  "fact": "один короткий интересный факт для ребёнка 6-10 лет, по-русски, не длиннее одного предложения"
}

Если на фото не камень и не ракушка (рука, мусор, песок, вода и т.п.) — верни category "other" и confidence 0.0.
Если ты не уверен, что именно это — всё равно дай наиболее вероятный вариант, но честно укажи низкий confidence.
Не придумывай наукообразных деталей, которых не видно на фото."""


class IdentifyResult(BaseModel):
    name: str
    category: str
    confidence: float
    fact: Optional[str] = None


def resize_for_upload(raw: bytes) -> bytes:
    """Downscale the image — we don't need megapixel photos for this prompt,
    and smaller images mean fewer tokens and a cheaper call."""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)
    return out.getvalue()


def parse_gemini_json(text: str) -> dict:
    """Gemini sometimes wraps JSON in ```json fences despite instructions — strip them."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def identify_photo(image_bytes: bytes) -> IdentifyResult:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            PROMPT,
        ],
    )
    try:
        data = parse_gemini_json(response.text)
        return IdentifyResult(**data)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        # Surface as a 502: it's Gemini's output that didn't match contract,
        # not a problem with the request itself. The app should treat this
        # the same as "no answer" — show the mystery-find card.
        raise HTTPException(
            status_code=502,
            detail=f"Gemini returned an unparseable response: {exc}",
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/identify", response_model=IdentifyResult)
async def identify(
    photo: UploadFile = File(...),
    x_device_id: str = Header(..., alias="X-Device-Id"),
    x_app_secret: Optional[str] = Header(None, alias="X-App-Secret"),
    x_is_pro: bool = Header(False, alias="X-Is-Pro"),
):
    # Simple shared-secret check so random clients can't hit this endpoint
    # and burn your Gemini quota. Fine for MVP; App Attest is the upgrade later.
    if APP_SHARED_SECRET and x_app_secret != APP_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="Invalid app secret")

    raw = await photo.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Photo too large")

    # The limit is enforced here, server-side — never trust the app's own
    # counter. x_is_pro comes from the app's receipt-validated subscription
    # state; it only skips the block, usage is still logged either way.
    usage_count = get_and_increment_usage(x_device_id)
    if not x_is_pro and usage_count > DAILY_FREE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Daily free limit of {DAILY_FREE_LIMIT} reached",
        )

    try:
        resized = resize_for_upload(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    return identify_photo(resized)