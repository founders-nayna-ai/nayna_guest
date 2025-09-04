# graph_router.py
# Component 1 — LangGraph Router (C1)
import os, json, re
from pathlib import Path
from typing import Any, Dict, List, TypedDict, Optional, Literal
from datetime import datetime, timedelta, timezone
import zoneinfo

import httpx
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, END

# -------------------- ENV / setup --------------------
load_dotenv()
app = FastAPI(title="Nayna – LangGraph Router (C1)")

ROUTER_TOKEN       = os.getenv("ROUTER_TOKEN", "")
DATAPULL_URL       = os.getenv("DATAPULL_URL", "")
DATAPULL_TOKEN     = os.getenv("DATAPULL_TOKEN", "")
META_TOKEN         = os.getenv("META_LONG_LIVED_TOKEN", "")
PHONE_ID           = os.getenv("PHONE_NUMBER_ID", "")
GRAPH_URL          = f"https://graph.facebook.com/v20.0/{PHONE_ID}/messages" if PHONE_ID else ""

OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
oai = OpenAI(api_key=OPENAI_API_KEY)

# show confidence footer to the user (toggle via .env SHOW_CONF=1)
SHOW_CONF          = os.getenv("SHOW_CONF", "1") == "1"

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
SESS_DIR = DATA_DIR / "sessions"; SESS_DIR.mkdir(exist_ok=True)
ONB_SESS_DIR = DATA_DIR / "onboarding_sessions"; ONB_SESS_DIR.mkdir(parents=True, exist_ok=True)
ROUTER_LOG = DATA_DIR / "router_log.jsonl"; ROUTER_LOG.touch(exist_ok=True)

# -------------------- time helpers --------------------
FALLBACK_TZ = timezone(timedelta(hours=5, minutes=30))  # IST

def safe_zoneinfo(tz_key: str):
    try:
        return zoneinfo.ZoneInfo(tz_key or "Asia/Kolkata")
    except Exception:
        return FALLBACK_TZ

def now_in_tz(tz_key: str) -> datetime:
    return datetime.now(safe_zoneinfo(tz_key))

def parse_sub_event_dt(wedding: Dict[str, Any], se: Dict[str, Any]) -> Optional[datetime]:
    tz = wedding.get("tz") or "Asia/Kolkata"
    try:
        y, m, d = map(int, (se.get("date") or "1900-01-01").split("-"))
        hh, mm = map(int, (se.get("time") or "00:00").split(":"))
    except Exception:
        return None
    return datetime(y, m, d, hh, mm, tzinfo=safe_zoneinfo(tz))

def current_or_next_sub_event(wedding: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return current/next sub-event by time; if all past, return the last."""
    subs = (wedding.get("sub_events") or [])
    if not subs:
        return None
    now = now_in_tz(wedding.get("tz"))
    timeline = []
    for se in subs:
        dt = parse_sub_event_dt(wedding, se)
        if dt:
            timeline.append((se, dt))
    if not timeline:
        return None
    timeline.sort(key=lambda x: x[1])
    for se, dt in timeline:
        if dt >= now:
            return se
    return timeline[-1][0]

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# -------------------- state schema --------------------
class RouterState(TypedDict, total=False):
    wa_id: str
    text: str
    source_event: Dict[str, Any]
    events: List[Dict[str, Any]]      # weddings + sub_events
    event_ids: List[str]
    reply: Optional[str]
    qa_result: Optional[Dict[str, Any]]
    onboard_result: Optional[Dict[str, Any]]
    route: Literal["onboard","query","done"]

# -------------------- utils --------------------
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def tokens(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(s))

def sess_path(wa_id: str) -> Path:
    return SESS_DIR / (re.sub(r"[^0-9A-Za-z_]+", "_", wa_id) + ".json")

def onb_sess_path(wa_id: str) -> Path:
    return ONB_SESS_DIR / (re.sub(r"[^0-9A-Za-z_]+", "_", wa_id) + ".json")

def load_session(wa_id: str) -> Dict[str, Any]:
    p = sess_path(wa_id)
    if not p.exists():
        return {
            "wa_id": wa_id,
            "history": [],
            "last_event_id": None,
            "pending_options": [],
            "last_user_msg_needing_answer": None
        }
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
        s.setdefault("pending_options", [])
        s.setdefault("last_user_msg_needing_answer", None)
        return s
    except Exception:
        return {
            "wa_id": wa_id, "history": [], "last_event_id": None,
            "pending_options": [], "last_user_msg_needing_answer": None
        }

def save_session(wa_id: str, sess: Dict[str, Any]) -> None:
    sess_path(wa_id).write_text(json.dumps(sess, ensure_ascii=False, indent=2), encoding="utf-8")

def load_onb_sess(wa_id:str) -> Dict[str, Any]:
    p = onb_sess_path(wa_id)
    if not p.exists(): return {"wa_id": wa_id, "state": "INIT", "data": {}, "created_at": now_iso()}
    try: return json.loads(p.read_text(encoding="utf-8"))
    except: return {"wa_id": wa_id, "state": "INIT", "data": {}, "created_at": now_iso()}

def save_onb_sess(wa_id:str, sess:Dict[str,Any])->None:
    onb_sess_path(wa_id).write_text(json.dumps(sess, ensure_ascii=False, indent=2), encoding="utf-8")

# greetings & acks
GREETING_WORDS = {"hi","hello","hey","yo","namaste","namaskar","salaam","hello ji","hii","hiii","heyy","sup"}
ACK_WORDS = {"ok","okay","okayy","okk","sure","done","thx","thanks","cool","great","acha","theek","haan","hmm","k","kk"}

def is_generic(text: str) -> bool:
    t = normalize_text(text)
    if len(t) <= 2:
        return True
    toks = set(tokens(t))
    return bool(toks & GREETING_WORDS)

# -------------------- WhatsApp sender --------------------
async def send_whatsapp_text(wa_id: str, message: str):
    if not (META_TOKEN and GRAPH_URL and wa_id and message):
        print("[C1] Missing WA creds or empty message")
        return
    headers = {"Authorization": f"Bearer {META_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": wa_id,
        "type": "text",
        "text": {"preview_url": False, "body": message[:4000]},
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(GRAPH_URL, headers=headers, json=payload)
        if r.status_code >= 300:
            print(f"[C1] WA send non-2xx: {r.status_code} {r.text}")

# -------------------- DataPull --------------------
async def call_datapull(wa_id: str) -> Dict[str, Any]:
    if not DATAPULL_URL:
        return {"ok": False, "events": [], "event_ids": []}
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

# -------------------- scoring & selection --------------------
MONTH_ALIASES = {
    "jan":"01","january":"01","feb":"02","february":"02","mar":"03","march":"03","apr":"04","april":"04",
    "may":"05","jun":"06","june":"06","jul":"07","july":"07","aug":"08","august":"08","sep":"09","sept":"09",
    "september":"09","oct":"10","october":"10","nov":"11","november":"11","dec":"12","december":"12"
}

def parse_date_hints(user_text: str) -> Dict[str, Any]:
    t = normalize_text(user_text)
    hints = {"today": ("today" in t) or ("aaj" in t),
             "tomorrow": ("tomorrow" in t) or ("tmrw" in t) or ("kal" in t),
             "month": None, "day": None}
    mday = re.search(r"\b([0-3]?\d)\b", t)
    if mday: hints["day"] = int(mday.group(1))
    for k, v in MONTH_ALIASES.items():
        if k in t: hints["month"] = int(v); break
    return hints

def score_wedding(user_text: str, w: Dict[str, Any], last_event_id: Optional[str]) -> float:
    """
    Selection score (0..1) based on:
      - token overlap with wedding_name/city/venue/id
      - token overlap with sub-event names
      - date hints (today/tomorrow/exact day-month)
      - stickiness if last_event_id matches
    """
    t = normalize_text(user_text)
    sc = 0.0
    fields = [("wedding_name",0.35),("city",0.20),("venue_name",0.20),("venue_address",0.10),("wedding_id",0.25)]
    for fname,wgt in fields:
        val = normalize_text(w.get(fname,""))
        if not val: continue
        hits = sum(1 for tok in set(tokens(t)) if tok and tok in val)
        if hits: sc += min(1.0, hits/4.0) * wgt
    sub_names = " ".join([normalize_text(se.get("event_name","")) for se in (w.get("sub_events") or [])])
    hits = sum(1 for tok in set(tokens(t)) if tok and tok in sub_names)
    if hits: sc += min(1.0, hits/3.0) * 0.25
    hints = parse_date_hints(user_text)
    nowd = now_in_tz(w.get("tz")).date()
    for se in (w.get("sub_events") or []):
        dt = parse_sub_event_dt(w, se)
        if not dt: continue
        d = dt.date()
        if hints["today"] and d == nowd: sc += 0.15
        if hints["tomorrow"] and d == (nowd + timedelta(days=1)): sc += 0.10
        if hints["month"] and hints["day"] and d.month == hints["month"] and d.day == hints["day"]:
            sc += 0.20; break
    if last_event_id and w.get("wedding_id") == last_event_id:
        sc += 0.20
    return max(0.0, min(sc, 1.0))

def pick_wedding(user_text: str, weddings: List[Dict[str, Any]], last_event_id: Optional[str]):
    if not weddings: return None, 0.0, []
    if len(weddings) == 1:
        eid = weddings[0]["wedding_id"]; return eid, 0.95, [(eid,0.95)]
    scored = [(w["wedding_id"], score_wedding(user_text, w, last_event_id)) for w in weddings]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0], scored[0][1], scored

# -------------------- sub-event selection helpers --------------------
BARAAT_ALIASES = {"baraat","barat","barat time","baarat","baaraat","ghodhi","ghodi","ghur charhai","sehra bandi"}
WEDDING_DAY_ALIASES = BARAAT_ALIASES | {"wedding","shaadi","shaadi day","pher","phere","pheras","phera","varmala","jaimala","jai mala","saat phere"}
MEHENDI_ALIASES = {"mehendi","mehandi","mehndi"}
SANGEET_ALIASES = {"sangeet"}
RECEPTION_ALIASES = {"reception"}
HALDI_ALIASES = {"haldi","tel ban","halad"}

def select_sub_event_for_query(user_msg: str, wedding: dict) -> Optional[dict]:
    """
    Map keywords → sub-event. If none match, fall back to current_or_next_sub_event().
    Special: any WEDDING_DAY_ALIASES -> choose a sub-event hinting wedding/phere/jaimala; else the last ceremony of the busiest wedding date.
    """
    t = normalize_text(user_msg)
    subs = wedding.get("sub_events") or []
    if not subs: return None

    def find_by_names(names: set[str]) -> Optional[dict]:
        for se in subs:
            se_name = normalize_text(se.get("event_name",""))
            for nm in names:
                if nm in se_name:
                    return se
        return None

    # hard maps
    if any(k in t for k in MEHENDI_ALIASES):
        se = find_by_names(MEHENDI_ALIASES)
        if se: return se
    if any(k in t for k in SANGEET_ALIASES):
        se = find_by_names(SANGEET_ALIASES)
        if se: return se
    if any(k in t for k in RECEPTION_ALIASES):
        se = find_by_names(RECEPTION_ALIASES)
        if se: return se
    if any(k in t for k in HALDI_ALIASES):
        se = find_by_names(HALDI_ALIASES)
        if se: return se

    # wedding/baraat/phere intent → wedding-day
    if any(k in t for k in WEDDING_DAY_ALIASES):
        se = find_by_names({"wedding","shaadi","phere","phera","pheras","varmala","jaimala","baraat","barat"})
        if se: return se
        # else: pick latest sub-event on the busiest date
        timeline = []
        for se in subs:
            dt = parse_sub_event_dt(wedding, se)
            if dt: timeline.append((se, dt))
        if timeline:
            timeline.sort(key=lambda x: x[1])
            dates = {}
            for se, dt in timeline:
                dates.setdefault(dt.date(), []).append((se, dt))
            best_date = max(dates.keys(), key=lambda d: (len(dates[d]), d))
            return sorted(dates[best_date], key=lambda x: x[1])[-1][0]

    # default safe fallback
    try:
        return current_or_next_sub_event(wedding)
    except NameError:
        # ultra-safe: if helper missing, pick the latest sub-event overall
        timeline = []
        for se in subs:
            dt = parse_sub_event_dt(wedding, se)
            if dt: timeline.append((se, dt))
        if not timeline:
            return None
        timeline.sort(key=lambda x: x[1])
        return timeline[-1][0]

# -------------------- intent & field detection --------------------
def list_sub_events(wedding: dict) -> list[dict]:
    return wedding.get("sub_events") or []

def union_sub_event_keys(wedding: dict) -> list[str]:
    keys=set()
    for se in list_sub_events(wedding):
        for k in se.keys(): keys.add(k)
    return sorted(keys)

def guess_user_fields(user_msg: str) -> List[str]:
    t = normalize_text(user_msg)
    want = []
    if any(w in t for w in ["kab","time","timing","start","konsi waqt","kitne baje"]): want.append("time")
    if any(w in t for w in ["date","din","konsi tareekh","which day"]): want.append("date")
    if any(w in t for w in ["venue","kahan","kahaan","kaha","where","location","hall","lawn","ballroom","poolside"]): want.append("venue_area")
    if any(w in t for w in ["dress code","dresscode","attire"]): want.append("dress_code")
    if any(w in t for w in ["theme"]): want.append("theme")
    if any(w in t for w in ["detail","kya hoga","plan","schedule","notes","program"]): want.append("notes")
    return want

def detect_intent(user_msg: str) -> str:
    t = normalize_text(user_msg)
    if any(w in t for w in GREETING_WORDS): return "greet"
    if any(w in t for w in ACK_WORDS): return "ack"
    if "link" in t or "links" in t or "url" in t: return "links"
    if ("dress" in t or "outfit" in t or "pehnu" in t or "pehen" in t or "look" in t) and \
       any(w in t for w in ["idea","ideas","suggest","suggestion","kya","koi","recommend","options","option"]):
        return "dress_suggest"
    if "suggest" in t or "ideas" in t or "recommend" in t or "kya lu" in t: return "suggest_generic"
    if any(w in t for w in ["menu","food","khana","dinner","lunch","snacks","cake"]): return "menu"
    if any(w in t for w in ["hotel","stay","room","route","reach","kaise pahuche","directions","parking"]): return "travel"
    if any(w in t for w in ["photo","pose","reel","camera"]): return "photo"
    if "gift" in t or "return gift" in t: return "gift"
    return "general"

# -------------------- formatting --------------------
def format_sub_event_card(wedding: dict, se: dict, asked_fields: List[str]) -> str:
    name_line = f"*{wedding.get('wedding_name','')}* — {wedding.get('city','')}"
    se_name = se.get("event_name","")
    rows = [name_line, f"*{se_name}:*"]
    def add(label, key):
        val = se.get(key)
        if val: rows.append(f"• *{label}:* {val}")
    if asked_fields:
        for f in asked_fields:
            if f == "date": add("Date","date")
            elif f == "time": add("Time","time")
            elif f == "venue_area": add("Venue","venue_area")
            elif f == "dress_code": add("Dress code","dress_code")
            elif f == "theme": add("Theme","theme")
            elif f == "notes": add("Notes","notes")
    else:
        add("Date","date"); add("Time","time")
    return "\n".join(rows)

def build_switch_footer(weddings: List[Dict[str, Any]], current_id: Optional[str]) -> str:
    if len(weddings) <= 1: return ""
    lines = ["\n—\n*Event badalna ho?* Number ya `event <ID>` bhejo:"]
    for idx, w in enumerate(weddings, start=1):
        tag = " (current)" if current_id and w.get("wedding_id")==current_id else ""
        lines.append(f"{idx}) {w.get('wedding_name')} — {w.get('city','')} [{w.get('wedding_id')}]"+tag)
    return "\n".join(lines)

# -------------------- LLM answering --------------------
def llm_answer(user_msg: str, wedding: dict, weddings_all: List[dict], current_id: Optional[str],
               ack: bool, score: Optional[float], add_footer: bool) -> str:
    intent = detect_intent(user_msg)
    asked_fields = guess_user_fields(user_msg)
    chosen = select_sub_event_for_query(user_msg, wedding) or {}
    card = format_sub_event_card(wedding, chosen, asked_fields)

    ctx = {
        "user_query": user_msg,
        "event_name": chosen.get("event_name",""),
        "city": wedding.get("city",""),
        "dress_code": chosen.get("dress_code",""),
        "theme": chosen.get("theme",""),
        "has_menu": bool(chosen.get("menu")),
    }

    system = (
        "You are Nayna — a playful Gen-Z Hinglish wedding helpdesk cousin.\n"
        "RULES:\n"
        "1) Never invent factual details. Facts come ONLY from the provided 'card'.\n"
        "2) Replies SHORT (1–2 lines) + max 2 emojis.\n"
        "3) If 'links' intent: say nicely you can't send links, but give 2–3 quick suggestions (no URLs).\n"
        "4) If suggestion intent (outfits/gifts/photos), give 2–3 practical ideas aligned to event_name/dress_code/theme/city.\n"
        "5) If info missing in card, say it's not available and offer a tiny helpful alternative.\n"
        "Tone: fun, supportive, Gen-Z, light Hinglish."
    )

    if intent == "ack":
        user_prompt = {"mode":"ack","context":ctx,"card":card}
    elif intent == "links":
        user_prompt = {"mode":"links","context":ctx,"card":card,"n_ideas":3}
    elif intent in {"dress_suggest","suggest_generic","photo","gift"}:
        user_prompt = {"mode":"suggest","context":ctx,"card":card,"n_ideas":3}
    elif intent == "menu":
        user_prompt = {"mode":"menu","context":ctx,"card":card}
    else:
        user_prompt = {"mode":"general","context":ctx,"card":card}

    r = oai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.55 if user_prompt["mode"] in {"suggest","links"} else 0.3,
        max_tokens=140,
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":json.dumps(user_prompt, ensure_ascii=False)}
        ]
    )
    head = r.choices[0].message.content.strip()

    show_card = bool(asked_fields) or user_prompt["mode"] in {"general","menu"}
    out = head + ("\n\n"+card if show_card else "")
    if add_footer and score is not None:
        out += f"\n\n_c-score: {score:.2f}_"
    out += build_switch_footer(weddings_all, current_id)
    return out

def llm_disambiguate(user_msg: str, options: List[Dict[str, Any]], scored: List[tuple[str,float]]) -> str:
    score_map = {eid: sc for eid, sc in scored}
    enriched = []
    for o in options:
        pct = int(round(100 * score_map.get(o["wedding_id"], 0.0)))
        enriched.append({"n": o["n"], "wedding_id": o["wedding_id"], "label": f'{o["label"]} – {pct}%'})
    ctx = {"user_query": user_msg, "options": enriched}
    system = (
        "You are Nayna, a concise Hinglish wedding helpdesk.\n"
        "Tell the user they’re linked to multiple weddings and ask which one.\n"
        "Show options exactly as provided (n, wedding_id, label). Friendly & short."
    )
    r = oai.chat.completions.create(
        model="gpt-4o-mini", temperature=0.3, max_tokens=220,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":json.dumps(ctx, ensure_ascii=False)}]
    )
    return r.choices[0].message.content.strip()

SWITCH_WORDS = ["change","switch","dusra","doosra","other","galat","wrong","different","badlo","badal","replace"]
def is_switch_intent(user_msg: str) -> bool:
    t = normalize_text(user_msg)
    if re.search(r"\bevent\s+[a-z0-9_-]+\b", t): return True
    return any(w in t for w in SWITCH_WORDS)

def parse_choice(user_msg: str, options: List[Dict[str, Any]]) -> Optional[str]:
    txt = normalize_text(user_msg)
    m = re.match(r"^\s*([1-9]\d*)\s*\.?\s*$", txt)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(options): return options[idx]["wedding_id"]
    for o in options:
        if o["wedding_id"].lower() in txt: return o["wedding_id"]
    for o in options:
        if normalize_text(o["label"]) in txt: return o["wedding_id"]
    return None

# -------------------- Onboarding (same) --------------------
def fuzzy_contains(needle: str, hay: str) -> bool:
    t = normalize_text(needle); h = normalize_text(hay)
    toks = [tok for tok in re.findall(r"[a-z0-9]+", t) if tok]
    if not toks: return False
    hits = sum(1 for tok in toks if tok in h)
    if len(toks) >= 2: return hits >= 2
    return hits >= 1

def find_event_by_code(db: Dict[str, Any], code: str) -> Optional[Dict[str, Any]]:
    code = (code or "").strip().upper()
    for w in db.get("weddings", []):
        if (w.get("wedding_id","").upper() == code): return w
    return None

EVENTS_DB_PATH = Path(os.getenv("EVENTS_DB", "data/events.json"))
def load_db() -> Dict[str, Any]:
    if not EVENTS_DB_PATH.exists(): return {"weddings": []}
    try: return json.loads(EVENTS_DB_PATH.read_text(encoding="utf-8"))
    except: return {"weddings": []}
def save_db(db: Dict[str, Any]) -> None:
    EVENTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVENTS_DB_PATH.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")

# -------------------- LangGraph nodes --------------------
async def node_datapull(state: RouterState) -> RouterState:
    wa_id = state["wa_id"]
    dp = await call_datapull(wa_id)
    weddings = dp.get("events", []) or []
    state["events"] = weddings
    state["event_ids"] = [w.get("wedding_id") for w in weddings]
    return state

def decide_route(state: RouterState) -> str:
    events = state.get("events") or []
    if not events: return "onboard"
    return "query"

async def node_onboard(state: RouterState) -> RouterState:
    wa_id = state["wa_id"]; text = state.get("text","")
    sess = load_onb_sess(wa_id); db = load_db()
    for w in db.get("weddings", []):
        for g in (w.get("guests") or []):
            if g.get("phone")==wa_id:
                state["reply"] = f"Aap pehle se onboard ho (event: {w.get('wedding_id')}). Apna sawal bhej sakte ho."
                state["route"] = "query"; return state
    st = sess.get("state","INIT")
    def ask_code():
        sess["state"]="ASK_CODE"; save_onb_sess(wa_id, sess)
        state["reply"] = "Aap naye lag rahe ho 🙂\nKripya apna *Event Code* bhej do (e.g., WED001)."
        state["route"]="done"; return state
    if st=="INIT": return ask_code()
    if st=="ASK_CODE":
        m = re.search(r"\b([A-Z]{3,}\d{2,})\b", text.strip(), re.IGNORECASE)
        if not m:
            state["reply"]="Event Code format e.g., WED001. Kripya sahi code bhejein."
            state["route"]="done"; return state
        code = m.group(1).upper()
        ev = find_event_by_code(db, code)
        if not ev:
            state["reply"]=f"Event Code *{code}* nahi mila. Sahi code bhejein."
            state["route"]="done"; return state
        sess["state"]="ASK_VERIFY"; sess["data"]["event_code"]=code; save_onb_sess(wa_id, sess)
        state["reply"]=("Thik hai! *{code}* mila.\n"
                        "Verification ke liye 2 chhote sawal:\n"
                        "1) Yeh kinki shaadi hai? (couple name)\n"
                        "2) City ka naam?\n"
                        "Dono ek message me bhej do (e.g., \"Sneha Rohit, Jaipur\").").format(code=code)
        state["route"]="done"; return state
    if st=="ASK_VERIFY":
        code = sess["data"].get("event_code"); ev = find_event_by_code(db, code)
        if not ev: return ask_code()
        parts = [p.strip() for p in re.split(r"[,\n;]+", text) if p.strip()]
        couple = parts[0] if parts else ""; city = parts[1] if len(parts)>1 else ""
        ok_c = fuzzy_contains(couple, ev.get("wedding_name","")); ok_city = fuzzy_contains(city, ev.get("city",""))
        if not (ok_c and ok_city):
            if not sess["data"].get("verify_retry"):
                sess["data"]["verify_retry"]=True; save_onb_sess(wa_id, sess)
                state["reply"]="Thoda match nahi hua. Couple name aur city dobara likh do (ek hi message me)."
                state["route"]="done"; return state
            sess["state"]="FAILED"; save_onb_sess(wa_id, sess)
            state["reply"]="Maaf kijiye, details match nahi hui. Filhaal aapko onboard nahi kar paaye."
            state["route"]="done"; return state
        sess["state"]="ASK_DETAILS"; sess["data"]["verified_event"]=code; save_onb_sess(wa_id, sess)
        state["reply"]="Great! Aap verified ho 👌\nAb apna *Name* aur *Relation* batayein (e.g., \"Rahul, Friend\")."
        state["route"]="done"; return state
    if st=="ASK_DETAILS":
        parts = [p.strip() for p in re.split(r"[,\n;]+", text) if p.strip()]
        name = parts[0] if parts else ""; relation = parts[1] if len(parts)>1 else ""
        if not name:
            state["reply"]="Apna *Name* aur *Relation* bhejein (e.g., \"Rahul, Friend\")."
            state["route"]="done"; return state
        code = sess["data"].get("verified_event")
        db = load_db(); ev = find_event_by_code(db, code)
        if not ev: return ask_code()
        guests = ev.get("guests") or []
        if not any(g.get("phone")==wa_id for g in guests):
            guests.append({"name": name, "phone": wa_id, "relation": relation})
            ev["guests"]=guests
            for i,w in enumerate(db.get("weddings",[])):
                if w.get("wedding_id")==code: db["weddings"][i]=ev; break
            save_db(db)
        sess["state"]="DONE"; save_onb_sess(wa_id, sess)
        state["reply"]=f"Welcome {name}! Aap *{ev.get('wedding_name')}* ({ev.get('city')}) me add ho gaye ho. Ab apna sawal bhejo."
        state["route"]="done"; return state
    if st=="FAILED":
        state["reply"]="Onboarding cancel ho gaya. Sahi details ke saath dobara try kar sakte ho."
        state["route"]="done"; return state
    return ask_code()

async def node_query(state: RouterState) -> RouterState:
    wa_id = state["wa_id"]; text = state.get("text",""); weddings = state.get("events") or []
    sess = load_session(wa_id)

    # explicit override by "event <ID>"
    m = re.search(r"\bevent\s+([A-Za-z0-9_-]+)\b", normalize_text(text))
    if m:
        eid = m.group(1).upper()
        sess["last_event_id"]=eid; sess["pending_options"]=[]
        sess["history"].append({"t":"override","event_id":eid,"msg":text,"at":now_iso()}); save_session(wa_id, sess)
        sel = next((w for w in weddings if (w.get("wedding_id","").upper()==eid)), None)
        if sel:
            reply = llm_answer(text, sel, weddings, eid, ack=True, score=1.0 if SHOW_CONF else None, add_footer=SHOW_CONF)
            state["reply"]=reply
            state["qa_result"]={"selected_event_id":eid,"needs_event_choice":False,"score":1.0}
            return state

    # resolve a pending choice → answer the ORIGINAL question that triggered disambiguation
    if sess.get("pending_options"):
        choice = parse_choice(text, sess["pending_options"])
        if choice:
            sel = next((w for w in weddings if w.get("wedding_id")==choice), None)
            if sel:
                original_q = sess.get("last_user_msg_needing_answer") or text
                sess["last_event_id"]=choice
                sess["pending_options"]=[]
                sess["last_user_msg_needing_answer"]=None
                sess["history"].append({"t":"choice","event_id":choice,"msg":text,"at":now_iso()})
                save_session(wa_id, sess)
                reply = llm_answer(original_q, sel, weddings, choice, ack=True,
                                   score=1.0 if SHOW_CONF else None, add_footer=SHOW_CONF)
                state["reply"]=reply
                state["qa_result"]={"selected_event_id":choice,"needs_event_choice":False,"score":1.0}
                return state

    # sticky, but skip if generic & multiple weddings (force re-ask)
    if sess.get("last_event_id") and not is_switch_intent(text) and not (is_generic(text) and len(weddings)>1):
        sel = next((w for w in weddings if w.get("wedding_id")==sess["last_event_id"]), None)
        if sel:
            sess["history"].append({"t":"sticky","event_id":sel["wedding_id"],"msg":text,"at":now_iso()}); save_session(wa_id, sess)
            reply = llm_answer(text, sel, weddings, sel["wedding_id"], ack=False,
                               score=1.0 if SHOW_CONF else None, add_footer=SHOW_CONF)
            state["reply"]=reply
            state["qa_result"]={"selected_event_id":sel["wedding_id"],"needs_event_choice":False,"score":1.0}
            return state

    # score + maybe disambiguate
    top_id, top_score, scored = pick_wedding(text, weddings, sess.get("last_event_id"))
    THRESHOLD=0.55; MARGIN=0.12
    if len(weddings)>1:
        need_choice = (top_score<THRESHOLD) or (len(scored)>=2 and (scored[0][1]-scored[1][1])<MARGIN) or is_generic(text)
        if need_choice:
            options=[{"n":i+1,"wedding_id":w["wedding_id"],"label":f"{w['wedding_name']} ({w.get('city','')})"} for i,w in enumerate(weddings)]
            sess["pending_options"]=options
            sess["last_user_msg_needing_answer"]=text
            sess["history"].append({"t":"ask","msg":text,"scores":scored,"at":now_iso()}); save_session(wa_id, sess)
            state["reply"]=llm_disambiguate(text, options, scored) + build_switch_footer(weddings, None)
            state["qa_result"]={"selected_event_id":None,"needs_event_choice":True,"options":options,"scores":scored}
            return state

    selected = next((w for w in weddings if w.get("wedding_id")==top_id), None) or (weddings[0] if weddings else None)
    if not selected:
        state["reply"]="Aap kis wedding ki baat kar rahe ho? Number ya event ID bhej do." + build_switch_footer(weddings, None)
        state["qa_result"]={"selected_event_id":None,"needs_event_choice":True}
        return state

    sess["last_event_id"]=selected["wedding_id"]; sess["pending_options"]=[]
    sess["last_user_msg_needing_answer"]=None
    sess["history"].append({"t":"msg","event_id":selected["wedding_id"],"msg":text,"score":top_score,"at":now_iso()})
    save_session(wa_id, sess)

    reply = llm_answer(text, selected, weddings, selected["wedding_id"], ack=True,
                       score=top_score if SHOW_CONF else None, add_footer=SHOW_CONF)
    state["reply"]=reply
    state["qa_result"]={"selected_event_id":selected["wedding_id"],"needs_event_choice":False,"score":top_score}
    return state

async def node_send_reply(state: RouterState) -> RouterState:
    if state.get("reply"):
        await send_whatsapp_text(state["wa_id"], state["reply"])
    return state

# -------------------- graph wiring --------------------
graph = StateGraph(RouterState)
graph.add_node("datapull", node_datapull)
graph.add_node("onboard", node_onboard)
graph.add_node("query", node_query)
graph.add_node("send_reply", node_send_reply)

graph.set_entry_point("datapull")
graph.add_conditional_edges("datapull", decide_route, {"onboard":"onboard","query":"query"})
graph.add_edge("onboard", "send_reply")
graph.add_edge("query", "send_reply")
graph.add_edge("send_reply", END)
compiled = graph.compile()

# -------------------- API --------------------
@app.post("/ingest")
async def ingest(request: Request, authorization: str | None = Header(default=None)):
    if ROUTER_TOKEN:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Missing bearer token")
        if authorization.split(" ", 1)[1] != ROUTER_TOKEN:
            raise HTTPException(403, "Invalid token")

    event = await request.json()
    wa_id = ((event.get("sender") or {}).get("wa_id")) or ""
    text = (event.get("message") or {}).get("text","") or ""
    if not wa_id:
        raise HTTPException(400, "wa_id missing")

    init: RouterState = {"wa_id": wa_id, "text": text, "source_event": event}
    out = await compiled.ainvoke(init)

    with ROUTER_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at":now_iso(),"wa_id":wa_id,"text":text,
                            "status":{"has_reply":bool(out.get("reply"))}}, ensure_ascii=False)+"\n")

    return JSONResponse({
        "ok": True,
        "qa": out.get("qa_result"),
        "onboard": out.get("onboard_result"),
        "sent": bool(out.get("reply"))
    })

@app.get("/")
def root(): return {"ok": True}
