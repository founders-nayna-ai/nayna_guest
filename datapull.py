import os, json
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="Nayna – Data Pulling Agent")

DATAPULL_TOKEN = os.getenv("DATAPULL_TOKEN", "")
WEDDINGS_DB    = Path(os.getenv("EVENTS_DB", "./data/events.json"))  # same env var; file now has "weddings"

def load_weddings() -> List[Dict[str, Any]]:
    if not WEDDINGS_DB.exists():
        return []
    try:
        data = json.loads(WEDDINGS_DB.read_text(encoding="utf-8"))
        return data.get("weddings", [])
    except Exception:
        return []

def find_weddings_for_contact(wa_id: str) -> List[Dict[str, Any]]:
    weddings = load_weddings()
    matched: List[Dict[str, Any]] = []
    for w in weddings:
        if wa_id in w.get("host_wa_ids", []):
            matched.append(w); continue
        for g in w.get("guests", []):
            if wa_id == g.get("phone"):
                matched.append(w); break
    return matched

@app.post("/pull")
async def pull(request: Request, authorization: str | None = Header(default=None)):
    # simple bearer auth so only C1 calls this
    if DATAPULL_TOKEN:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Missing bearer token")
        token = authorization.split(" ", 1)[1]
        if token != DATAPULL_TOKEN:
            raise HTTPException(403, "Invalid token")

    body = await request.json()
    wa_id: str = (body.get("wa_id") or "").strip()
    if not wa_id:
        raise HTTPException(400, "wa_id is required")

    weddings = find_weddings_for_contact(wa_id)
    wedding_ids = [w["wedding_id"] for w in weddings]

    return JSONResponse({
        "ok": True,
        "wa_id": wa_id,
        "event_ids": wedding_ids,   # keeps name from old contract so C1/QA continue to work
        "events": weddings          # full wedding objects incl. sub_events
    })

@app.get("/")
def root(): return {"ok": True}
