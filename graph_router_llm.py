# graph_router_llm.py
# LLM-Driven LangGraph Router
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
app = FastAPI(title="Nayna – LLM-Driven LangGraph Router")

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
    # LLM decision results
    llm_analysis: Optional[Dict[str, Any]]
    selected_wedding: Optional[Dict[str, Any]]
    selected_sub_event: Optional[Dict[str, Any]]
    user_intent: Optional[str]
    needs_clarification: bool

# -------------------- LLM Decision Functions --------------------

def llm_analyze_user_intent(user_msg: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Use LLM to analyze user intent and classify the message"""
    from prompt_templates import PromptTemplates
    
    # Enhanced context analysis
    enhanced_context = {
        **context,
        "message_length": len(user_msg),
        "has_emojis": any(ord(char) > 127 for char in user_msg),
        "language_detected": "hinglish" if any(word in user_msg.lower() for word in ["aap", "tum", "hai", "hain", "ka", "ki", "ko"]) else "english"
    }
    
    system_prompt = PromptTemplates.get_intent_analysis_prompt()
    user_prompt = f"""User message: "{user_msg}"

Context:
- Wedding count: {enhanced_context.get('wedding_count', 0)}
- Has history: {enhanced_context.get('has_history', False)}
- Last event: {enhanced_context.get('last_event', 'None')}
- Message length: {enhanced_context.get('message_length', 0)} characters
- Contains emojis: {enhanced_context.get('has_emojis', False)}
- Language: {enhanced_context.get('language_detected', 'unknown')}"""

    try:
        response = oai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=400,
            messages=[
                {"role": "system", "content": system_prompt.format(**enhanced_context)},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        return result
    except Exception as e:
        return {
            "intent": "query",
            "confidence": 0.5,
            "reasoning": f"Error in analysis: {str(e)}",
            "needs_clarification": False,
            "suggested_response": "",
            "extracted_entities": {},
            "emotional_tone": "neutral",
            "urgency_level": "low"
        }

def llm_select_wedding(user_msg: str, weddings: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
    """Use LLM to select the most relevant wedding based on user message"""
    from prompt_templates import PromptTemplates
    
    if not weddings:
        return {"selected_wedding": None, "confidence": 0.0, "reasoning": "No weddings available"}
    
    if len(weddings) == 1:
        return {"selected_wedding": weddings[0], "confidence": 1.0, "reasoning": "Only one wedding available"}
    
    # Prepare comprehensive wedding summaries for LLM
    wedding_summaries = []
    for i, wedding in enumerate(weddings):
        summary = {
            "index": i + 1,
            "wedding_id": wedding.get("wedding_id", ""),
            "wedding_name": wedding.get("wedding_name", ""),
            "city": wedding.get("city", ""),
            "venue_name": wedding.get("venue_name", ""),
            "venue_address": wedding.get("venue_address", ""),
            "sub_events": [se.get("event_name", "") for se in wedding.get("sub_events", [])],
            "hosts": [g.get("name", "") for g in wedding.get("guests", []) if g.get("relation") == "host"],
            "all_guests": [{"name": g.get("name", ""), "relation": g.get("relation", "")} for g in wedding.get("guests", [])]
        }
        wedding_summaries.append(summary)
    
    system_prompt = PromptTemplates.get_wedding_selection_prompt()
    user_prompt = f"""User message: "{user_msg}"

Available weddings:
{json.dumps(wedding_summaries, indent=2)}

Context:
- Last selected wedding: {context.get('last_wedding_id', 'None')}
- User history: {context.get('has_history', False)}
- User intent: {context.get('user_intent', 'query')}"""

    try:
        response = oai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=500,
            messages=[
                {"role": "system", "content": system_prompt.format(wedding_count=len(weddings))},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        
        # Get the selected wedding
        selected_index = result.get("selected_index", 1) - 1
        if 0 <= selected_index < len(weddings):
            result["selected_wedding"] = weddings[selected_index]
        else:
            result["selected_wedding"] = weddings[0]
            result["confidence"] = 0.5
            result["reasoning"] = "Invalid selection, using first wedding"
        
        return result
    except Exception as e:
        return {
            "selected_wedding": weddings[0],
            "confidence": 0.5,
            "reasoning": f"Error in selection: {str(e)}",
            "needs_clarification": True,
            "clarification_question": "Which wedding are you asking about?",
            "extracted_mentions": {}
        }

def llm_select_sub_event(user_msg: str, wedding: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Use LLM to select the most relevant sub-event within a wedding"""
    from prompt_templates import PromptTemplates
    
    sub_events = wedding.get("sub_events", [])
    if not sub_events:
        return {"selected_sub_event": None, "confidence": 0.0, "reasoning": "No sub-events available"}
    
    if len(sub_events) == 1:
        return {"selected_sub_event": sub_events[0], "confidence": 1.0, "reasoning": "Only one sub-event available"}
    
    # Prepare comprehensive sub-event summaries
    sub_event_summaries = []
    for i, se in enumerate(sub_events):
        summary = {
            "index": i + 1,
            "event_name": se.get("event_name", ""),
            "date": se.get("date", ""),
            "time": se.get("time", ""),
            "venue_area": se.get("venue_area", ""),
            "dress_code": se.get("dress_code", ""),
            "theme": se.get("theme", ""),
            "notes": se.get("notes", ""),
            "menu": se.get("menu", ""),
            "special_instructions": se.get("special_instructions", "")
        }
        sub_event_summaries.append(summary)
    
    system_prompt = PromptTemplates.get_sub_event_selection_prompt()
    user_prompt = f"""User message: "{user_msg}"

Wedding: {wedding.get('wedding_name', '')} - {wedding.get('city', '')}

Available sub-events:
{json.dumps(sub_event_summaries, indent=2)}

Context:
- Current time: {context.get('current_time', 'Not specified')}
- User intent: {context.get('user_intent', 'Not specified')}
- Wedding context: {context.get('wedding_context', 'Not specified')}"""

    try:
        response = oai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=500,
            messages=[
                {"role": "system", "content": system_prompt.format(sub_event_count=len(sub_events))},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        
        # Get the selected sub-event
        selected_index = result.get("selected_index", 1) - 1
        if 0 <= selected_index < len(sub_events):
            result["selected_sub_event"] = sub_events[selected_index]
        else:
            result["selected_sub_event"] = sub_events[0]
            result["confidence"] = 0.5
            result["reasoning"] = "Invalid selection, using first sub-event"
        
        return result
    except Exception as e:
        return {
            "selected_sub_event": sub_events[0],
            "confidence": 0.5,
            "reasoning": f"Error in selection: {str(e)}",
            "needs_clarification": True,
            "clarification_question": "Which event are you asking about?",
            "extracted_mentions": {}
        }

def llm_generate_response(user_msg: str, context: Dict[str, Any]) -> str:
    """Use LLM to generate a contextual response"""
    from prompt_templates import PromptTemplates
    
    system_prompt = PromptTemplates.get_response_generation_prompt()
    
    # Enhanced context for better responses
    enhanced_context = {
        **context,
        "user_message": user_msg,
        "response_style": context.get('intent', 'query'),
        "has_event_details": bool(context.get('event_details')),
        "needs_help": context.get('needs_clarification', False),
        "emotional_tone": context.get('emotional_tone', 'neutral'),
        "urgency": context.get('urgency_level', 'low')
    }
    
    user_prompt = f"""User message: "{user_msg}"

Context:
- Intent: {enhanced_context.get('intent', 'Unknown')}
- Selected wedding: {enhanced_context.get('wedding_name', 'None')}
- Selected event: {enhanced_context.get('event_name', 'None')}
- Event details: {json.dumps(enhanced_context.get('event_details', {}), indent=2)}
- Needs clarification: {enhanced_context.get('needs_clarification', False)}
- Clarification question: {enhanced_context.get('clarification_question', 'None')}
- Emotional tone: {enhanced_context.get('emotional_tone', 'neutral')}
- Urgency: {enhanced_context.get('urgency_level', 'low')}

Generate a response that:
1. Acknowledges the user's message appropriately
2. Provides relevant information if available
3. Asks for clarification if needed
4. Maintains the fun, helpful personality
5. Matches the user's emotional tone
6. Addresses any urgency appropriately"""

    try:
        response = oai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=250,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Hey! I'm having a small technical issue right now 😅 Can you try asking again?"

# -------------------- utils --------------------
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def sess_path(wa_id: str) -> Path:
    return SESS_DIR / (re.sub(r"[^0-9A-Za-z_]+", "_", wa_id) + ".json")

def load_session(wa_id: str) -> Dict[str, Any]:
    p = sess_path(wa_id)
    if not p.exists():
        return {
            "wa_id": wa_id,
            "history": [],
            "last_wedding_id": None,
            "last_sub_event_id": None,
            "conversation_state": "initial"
        }
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
        s.setdefault("conversation_state", "initial")
        return s
    except Exception:
        return {
            "wa_id": wa_id, "history": [], "last_wedding_id": None,
            "last_sub_event_id": None, "conversation_state": "initial"
        }

def save_session(wa_id: str, sess: Dict[str, Any]) -> None:
    sess_path(wa_id).write_text(json.dumps(sess, ensure_ascii=False, indent=2), encoding="utf-8")

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
    if not events: 
        return "onboard"
    return "analyze"

async def node_analyze(state: RouterState) -> RouterState:
    """LLM-driven analysis of user intent and context"""
    wa_id = state["wa_id"]
    text = state.get("text", "")
    events = state.get("events", [])
    
    # Load session context
    sess = load_session(wa_id)
    
    # Prepare context for LLM
    context = {
        "wedding_count": len(events),
        "has_history": len(sess.get("history", [])) > 0,
        "last_event": sess.get("last_wedding_id"),
        "current_time": now_iso()
    }
    
    # Analyze user intent
    intent_analysis = llm_analyze_user_intent(text, context)
    state["llm_analysis"] = intent_analysis
    state["user_intent"] = intent_analysis.get("intent", "query")
    state["needs_clarification"] = intent_analysis.get("needs_clarification", False)
    
    # If needs clarification, go to clarification
    if intent_analysis.get("needs_clarification"):
        return "clarify"
    
    # If greeting or general chat, go to response
    if intent_analysis.get("intent") in ["greeting", "general_chat"]:
        return "respond"
    
    # If onboarding needed, go to onboard
    if intent_analysis.get("intent") == "onboarding":
        return "onboard"
    
    # Otherwise, proceed with query processing
    return "query"

async def node_query(state: RouterState) -> RouterState:
    """LLM-driven query processing"""
    wa_id = state["wa_id"]
    text = state.get("text", "")
    events = state.get("events", [])
    sess = load_session(wa_id)
    
    # Select wedding using LLM
    wedding_context = {
        "last_wedding_id": sess.get("last_wedding_id"),
        "has_history": len(sess.get("history", [])) > 0
    }
    
    wedding_selection = llm_select_wedding(text, events, wedding_context)
    state["selected_wedding"] = wedding_selection.get("selected_wedding")
    
    if not state["selected_wedding"]:
        state["needs_clarification"] = True
        state["reply"] = "I couldn't figure out which wedding you're asking about. Can you tell me the wedding name or city?"
        return state
    
    # Select sub-event using LLM
    sub_event_context = {
        "current_time": now_iso(),
        "user_intent": state.get("user_intent", "query")
    }
    
    sub_event_selection = llm_select_sub_event(text, state["selected_wedding"], sub_event_context)
    state["selected_sub_event"] = sub_event_selection.get("selected_sub_event")
    
    # Generate response
    response_context = {
        "intent": state.get("user_intent", "query"),
        "wedding_name": state["selected_wedding"].get("wedding_name", ""),
        "event_name": state["selected_sub_event"].get("event_name", "") if state["selected_sub_event"] else "",
        "event_details": state["selected_sub_event"] or {},
        "needs_clarification": sub_event_selection.get("needs_clarification", False),
        "clarification_question": sub_event_selection.get("clarification_question", "")
    }
    
    state["reply"] = llm_generate_response(text, response_context)
    
    # Update session
    sess["last_wedding_id"] = state["selected_wedding"].get("wedding_id")
    sess["last_sub_event_id"] = state["selected_sub_event"].get("event_name") if state["selected_sub_event"] else None
    sess["history"].append({
        "timestamp": now_iso(),
        "user_msg": text,
        "intent": state.get("user_intent"),
        "wedding_id": state["selected_wedding"].get("wedding_id"),
        "sub_event": state["selected_sub_event"].get("event_name") if state["selected_sub_event"] else None
    })
    save_session(wa_id, sess)
    
    return state

async def node_clarify(state: RouterState) -> RouterState:
    """Handle clarification requests"""
    analysis = state.get("llm_analysis", {})
    state["reply"] = analysis.get("suggested_response", "Can you please clarify what you're looking for?")
    return state

async def node_respond(state: RouterState) -> RouterState:
    """Handle greetings and general chat"""
    text = state.get("text", "")
    context = {
        "intent": state.get("user_intent", "greeting"),
        "wedding_count": len(state.get("events", [])),
        "has_history": False
    }
    state["reply"] = llm_generate_response(text, context)
    return state

async def node_onboard(state: RouterState) -> RouterState:
    """LLM-driven onboarding process"""
    from llm_onboarding import LLMOnboardingSystem
    
    wa_id = state["wa_id"]
    text = state.get("text", "")
    events = state.get("events", [])
    
    # Initialize LLM onboarding system
    onboarding_system = LLMOnboardingSystem(oai)
    
    # Process onboarding message
    result = await onboarding_system.process_onboarding_message(wa_id, text, events)
    
    state["reply"] = result["reply"]
    state["onboard_result"] = {
        "onboarded": result["onboarded"],
        "forward_to_query": result["forward_to_query"],
        "session_state": result["session_state"]
    }
    
    return state

async def node_send_reply(state: RouterState) -> RouterState:
    if state.get("reply"):
        await send_whatsapp_text(state["wa_id"], state["reply"])
    return state

# -------------------- graph wiring --------------------
graph = StateGraph(RouterState)
graph.add_node("datapull", node_datapull)
graph.add_node("analyze", node_analyze)
graph.add_node("query", node_query)
graph.add_node("clarify", node_clarify)
graph.add_node("respond", node_respond)
graph.add_node("onboard", node_onboard)
graph.add_node("send_reply", node_send_reply)

graph.set_entry_point("datapull")
graph.add_conditional_edges("datapull", decide_route, {"onboard":"onboard","analyze":"analyze"})
graph.add_conditional_edges("analyze", lambda state: state.get("user_intent", "query"), {
    "onboarding": "onboard",
    "greeting": "respond", 
    "general_chat": "respond",
    "clarification": "clarify",
    "query": "query",
    "switch_event": "query"
})
graph.add_edge("query", "send_reply")
graph.add_edge("clarify", "send_reply")
graph.add_edge("respond", "send_reply")
graph.add_edge("onboard", "send_reply")
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
