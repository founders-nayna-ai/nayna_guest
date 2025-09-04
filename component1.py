import os, json
from typing import Dict, Any
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pathlib import Path
import httpx
from datetime import datetime, timezone

load_dotenv()
app = FastAPI(title="Nayna – Component 1 (Master Router)")

# ---- Auth & internal URLs ----
ROUTER_TOKEN       = os.getenv("ROUTER_TOKEN", "")

DATAPULL_URL       = os.getenv("DATAPULL_URL", "")
DATAPULL_TOKEN     = os.getenv("DATAPULL_TOKEN", "")

QUERY_AGENTS_URL   = os.getenv("QUERY_AGENTS_URL", "")
QUERY_AGENTS_TOKEN = os.getenv("QUERY_AGENTS_TOKEN", "")

# NEW: Onboarding Agent
ONBOARDING_URL     = os.getenv("ONBOARDING_URL", "http://localhost:8400/onboard")
ONBOARDING_TOKEN   = os.getenv("ONBOARDING_TOKEN", "")

# ---- WhatsApp send (same creds as C0) ----
META_TOKEN   = os.getenv("META_LONG_LIVED_TOKEN", "")
PHONE_ID     = os.getenv("PHONE_NUMBER_ID", "")
GRAPH_URL    = f"https://graph.facebook.com/v20.0/{PHONE_ID}/messages" if PHONE_ID else ""

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
ROUTER_LOG = DATA_DIR / "router_log.jsonl"
if not ROUTER_LOG.exists(): ROUTER_LOG.touch()

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ---------- helpers ----------

async def call_datapull(wa_id: str) -> Dict[str, Any]:
    if not DATAPULL_URL:
        return {"ok": False, "error": "DATAPULL_URL not set", "events": [], "event_ids": []}
    headers = {"Content-Type": "application/json"}
    if DATAPULL_TOKEN:
        headers["Authorization"] = f"Bearer {DATAPULL_TOKEN}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(DATAPULL_URL, headers=headers, json={"wa_id": wa_id})
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        return {"ok": False, "error": str(e), "events": [], "event_ids": []}

async def call_query_agent(packet: dict) -> Dict[str, Any]:
    if not QUERY_AGENTS_URL:
        with ROUTER_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"no_QA": True, "packet": packet}, ensure_ascii=False) + "\n")
        return {"ok": False, "error": "QUERY_AGENTS_URL not set"}
    headers = {"Content-Type": "application/json"}
    if QUERY_AGENTS_TOKEN:
        headers["Authorization"] = f"Bearer {QUERY_AGENTS_TOKEN}"
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(QUERY_AGENTS_URL, headers=headers, json=packet)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        with ROUTER_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"qa_error": str(e), "packet": packet}, ensure_ascii=False) + "\n")
        return {"ok": False, "error": str(e)}

async def call_onboarding(wa_id: str, text: str) -> Dict[str, Any]:
    if not ONBOARDING_URL:
        return {"onboarded": False, "reply": "Onboarding service not configured.", "forward_to_query": False}
    headers = {"Content-Type": "application/json"}
    if ONBOARDING_TOKEN:
        headers["Authorization"] = f"Bearer {ONBOARDING_TOKEN}"
    payload = {"wa_id": wa_id, "text": text}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(ONBOARDING_URL, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        return {"onboarded": False, "reply": f"Onboarding error: {e}", "forward_to_query": False}

async def send_whatsapp_text(wa_id: str, message: str):
    if not (META_TOKEN and GRAPH_URL):
        print("[C1] META creds missing; cannot send WA message.")
        return
    headers = {
        "Authorization": f"Bearer {META_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": wa_id,
        "type": "text",
        "text": {"preview_url": False, "body": (message or "")[:4000]},
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(GRAPH_URL, headers=headers, json=payload)
            if r.status_code >= 300:
                print(f"[C1] WA send non-2xx: {r.status_code} {r.text}")
    except httpx.HTTPError as e:
        print(f"[C1] WA send error: {e}")

# ---------- main route ----------

@app.post("/ingest")
async def ingest(request: Request, authorization: str | None = Header(default=None)):
    # verify C0
    if ROUTER_TOKEN:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Missing bearer token")
        token = authorization.split(" ", 1)[1]
        if token != ROUTER_TOKEN:
            raise HTTPException(403, "Invalid token")

    event = await request.json()
    wa_id = ((event.get("sender") or {}).get("wa_id")) or ""
    message = event.get("message") or {}
    text = message.get("text", "") or ""  # for now we handle only text

    # 1) DataPull: find weddings for this contact
    dp = await call_datapull(wa_id)
    weddings = dp.get("events", []) or []
    event_ids = [w.get("wedding_id") for w in weddings]

    # If NO events → run Onboarding
    if not weddings:
        ob = await call_onboarding(wa_id, text)
        ob_reply = ob.get("reply")
        if ob_reply:
            await send_whatsapp_text(wa_id, ob_reply)

        # If onboarded, refresh DataPull once and forward to Query Agent
        if ob.get("onboarded") and ob.get("forward_to_query"):
            dp2 = await call_datapull(wa_id)
            weddings2 = dp2.get("events", []) or []
            event_ids2 = [w.get("wedding_id") for w in weddings2]

            packet2 = {
                "version": "1.0",
                "routed_at": now_iso(),
                "source_event": event,
                "contact": {"wa_id": wa_id, "event_ids": event_ids2},
                "event_context": weddings2
            }
            qa2 = await call_query_agent(packet2)
            reply2 = qa2.get("reply") if isinstance(qa2, dict) else None
            if reply2:
                await send_whatsapp_text(wa_id, reply2)

            print(f"[C1] NEW USER onboarded wa_id={wa_id} | events={event_ids2} | text={text!r} | qa_reply? {bool(reply2)}")
            return JSONResponse({"ok": True, "event_ids": event_ids2, "qa_ok": bool(reply2), "onboarded": True})

        # Not onboarded → stop here
        print(f"[C1] NEW USER pending/failed onboarding wa_id={wa_id} | text={text!r}")
        return JSONResponse({"ok": True, "event_ids": [], "qa_ok": False, "onboarded": False})

    # 2) Build packet for Query Agent (known contact)
    packet = {
        "version": "1.0",
        "routed_at": now_iso(),
        "source_event": event,        # normalized C0 event
        "contact": {"wa_id": wa_id, "event_ids": event_ids},
        "event_context": weddings     # full wedding objects (with sub_events)
    }

    # 3) Call Query Agent (LLM + logic)
    qa = await call_query_agent(packet)
    reply = qa.get("reply") if isinstance(qa, dict) else None

    print(f"[C1] wa_id={wa_id} | events={event_ids} | text={text!r} | reply? {bool(reply)}")

    # 4) Auto-send reply to WhatsApp (if available)
    if reply:
        await send_whatsapp_text(wa_id, reply)
    else:
        with ROUTER_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"no_reply": True, "packet": packet, "qa": qa}, ensure_ascii=False) + "\n")

    return JSONResponse({"ok": True, "event_ids": event_ids, "qa_ok": bool(reply)})

@app.get("/")
def root(): return {"ok": True}
