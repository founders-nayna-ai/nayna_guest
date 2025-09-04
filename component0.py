import os, json, hmac, hashlib
from typing import Any, Dict, Optional
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
import httpx
from pathlib import Path
from datetime import datetime, timezone

# -------- setup --------
load_dotenv()
app = FastAPI(title="Nayna – Component 0 (Gateway)")

VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "")
APP_SECRET   = os.getenv("META_APP_SECRET", "")
ROUTER_URL   = os.getenv("ROUTER_URL", "")
ROUTER_TOKEN = os.getenv("ROUTER_TOKEN", "")

DATA_DIR  = Path("data"); DATA_DIR.mkdir(exist_ok=True)
LOG_FILE  = DATA_DIR / "messages_log.jsonl"
SEEN_FILE = DATA_DIR / "seen_ids.json"
if not LOG_FILE.exists():  LOG_FILE.touch()
if not SEEN_FILE.exists(): SEEN_FILE.write_text(json.dumps({"ids": []}), encoding="utf-8")

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def verify_signature(app_secret: str, payload: bytes, signature_header: Optional[str]) -> bool:
    if not signature_header or not signature_header.startswith("sha256="): return False
    sent = signature_header.split("=", 1)[1]
    mac = hmac.new(app_secret.encode("utf-8"), msg=payload, digestmod=hashlib.sha256)
    return hmac.compare_digest(sent, mac.hexdigest())

def store_full_payload(payload: dict) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def mark_and_check_duplicate(message_id: Optional[str], max_keep: int = 5000) -> bool:
    if not message_id: return False
    data = json.loads(SEEN_FILE.read_text(encoding="utf-8"))
    seen = data.get("ids", [])
    if message_id in seen: return True
    seen.append(message_id)
    if len(seen) > max_keep: seen = seen[-max_keep:]
    SEEN_FILE.write_text(json.dumps({"ids": seen}), encoding="utf-8")
    return False

async def forward_to_router(event: dict) -> None:
    if not ROUTER_URL:
        print("[C0] ROUTER_URL not set; skipping forward."); return
    headers = {"Content-Type": "application/json"}
    if ROUTER_TOKEN: headers["Authorization"] = f"Bearer {ROUTER_TOKEN}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(ROUTER_URL, headers=headers, json=event)
            if r.status_code >= 300:
                print(f"[C0] Router POST non-2xx: {r.status_code} {r.text}")
    except httpx.HTTPError as e:
        print(f"[C0] Router unreachable ({ROUTER_URL}): {e}")

@app.get("/webhook")
async def verify(
    hub_mode: str = Query(alias="hub.mode"),
    hub_challenge: str = Query(alias="hub.challenge"),
    hub_verify_token: str = Query(alias="hub.verify_token")
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return PlainTextResponse(content=hub_challenge, status_code=200)
    raise HTTPException(status_code=403, detail="Verification failed")

@app.post("/webhook")
async def webhook(request: Request):
    body = await request.body()
    sig  = request.headers.get("X-Hub-Signature-256")

    # 1) Verify
    if not verify_signature(APP_SECRET, body, sig):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # 2) Parse
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # 3) Store raw payload
    store_full_payload(payload)

    # 4) Normalize only text; de-dupe right after storing
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value: Dict[str, Any] = change.get("value", {})
            messages = value.get("messages", [])
            contacts = value.get("contacts", [])
            wa_id = contacts[0].get("wa_id") if contacts else None

            for msg in messages:
                msg_id = msg.get("id")
                if mark_and_check_duplicate(msg_id):  # skip repeats
                    continue
                if msg.get("type") != "text":
                    continue

                text = (msg.get("text") or {}).get("body", "")
                ts   = msg.get("timestamp") or now_iso()
                event = {
                    "source": "whatsapp",
                    "version": "1.0",
                    "received_at": now_iso(),
                    "sender": {"wa_id": wa_id},
                    "message": {"id": msg_id, "type": "text", "text": text, "timestamp": ts},
                    "context": {
                        "meta_phone_number_id": value.get("metadata", {}).get("phone_number_id"),
                        "profile": (contacts[0].get("profile") if contacts else None),
                    },
                }
                print(f"[C0] -> C1 {event['message']['id']} {event['message']['text']!r}")
                await forward_to_router(event)

    # Always ack Meta (avoid retries)
    return {"success": True}

@app.get("/")
def root(): return {"ok": True}

@app.get("/health")
def health(): return {"status": "up"}
