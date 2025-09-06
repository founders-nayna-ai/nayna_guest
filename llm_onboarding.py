# llm_onboarding.py
# LLM-Driven Onboarding System
import json
import re
from typing import Dict, Any, Optional
from openai import OpenAI
from pathlib import Path

class LLMOnboardingSystem:
    def __init__(self, openai_client: OpenAI):
        self.oai = openai_client
        self.onb_sess_dir = Path("data/onboarding_sessions")
        self.onb_sess_dir.mkdir(parents=True, exist_ok=True)
    
    def onb_sess_path(self, wa_id: str) -> Path:
        return self.onb_sess_dir / (re.sub(r"[^0-9A-Za-z_]+", "_", wa_id) + ".json")
    
    def load_onb_sess(self, wa_id: str) -> Dict[str, Any]:
        p = self.onb_sess_path(wa_id)
        if not p.exists():
            return {
                "wa_id": wa_id,
                "state": "INIT",
                "data": {},
                "created_at": self.now_iso(),
                "conversation_history": []
            }
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except:
            return {
                "wa_id": wa_id,
                "state": "INIT",
                "data": {},
                "created_at": self.now_iso(),
                "conversation_history": []
            }
    
    def save_onb_sess(self, wa_id: str, sess: Dict[str, Any]) -> None:
        self.onb_sess_path(wa_id).write_text(
            json.dumps(sess, ensure_ascii=False, indent=2), 
            encoding="utf-8"
        )
    
    def now_iso(self) -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
    
    def llm_analyze_onboarding_step(self, user_msg: str, session: Dict[str, Any], available_events: list) -> Dict[str, Any]:
        """Use LLM to determine the next step in onboarding"""
        
        system_prompt = """You are Nayna, an AI wedding assistant helping new users get onboarded.

ONBOARDING STATES:
- INIT: First interaction, need to get event code or wedding info
- VERIFYING: Verifying event code and user details
- COLLECTING_DETAILS: Getting user's name and relation
- COMPLETED: Onboarding finished
- FAILED: Onboarding failed, need to restart

ANALYSIS CRITERIA:
1. Look for event codes (format: 3+ letters + 2+ numbers, e.g., WED001, ABC123)
2. Look for wedding names, couple names, cities
3. Look for user details (name, relation to couple)
4. Determine if verification is needed
5. Check if onboarding can be completed

Respond with JSON:
{
    "next_state": "one_of_the_states_above",
    "confidence": 0.0-1.0,
    "reasoning": "explanation of decision",
    "extracted_data": {
        "event_code": "if found",
        "wedding_name": "if mentioned",
        "couple_name": "if mentioned", 
        "city": "if mentioned",
        "user_name": "if mentioned",
        "relation": "if mentioned"
    },
    "needs_clarification": true/false,
    "clarification_question": "if clarification needed",
    "suggested_response": "response to send to user"
}"""

        # Prepare context
        conversation_history = session.get("conversation_history", [])
        current_state = session.get("state", "INIT")
        session_data = session.get("data", {})
        
        # Create event summaries for context
        event_summaries = []
        for event in available_events:
            summary = {
                "wedding_id": event.get("wedding_id", ""),
                "wedding_name": event.get("wedding_name", ""),
                "city": event.get("city", ""),
                "couple_names": [g.get("name", "") for g in event.get("guests", []) if g.get("relation") == "host"]
            }
            event_summaries.append(summary)
        
        user_prompt = f"""User message: "{user_msg}"

Current session state: {current_state}
Session data: {json.dumps(session_data, indent=2)}
Conversation history: {json.dumps(conversation_history[-3:], indent=2)}  # Last 3 messages

Available events for verification:
{json.dumps(event_summaries, indent=2)}

Analyze this onboarding step and determine what to do next."""

        try:
            response = self.oai.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            return result
        except Exception as e:
            return {
                "next_state": "INIT",
                "confidence": 0.5,
                "reasoning": f"Error in analysis: {str(e)}",
                "extracted_data": {},
                "needs_clarification": True,
                "clarification_question": "I'm having trouble understanding. Can you tell me your event code?",
                "suggested_response": "Hey! I'm having a small technical issue. Can you share your event code?"
            }
    
    def llm_verify_event_match(self, extracted_data: Dict[str, Any], available_events: list) -> Dict[str, Any]:
        """Use LLM to verify if extracted data matches any available events"""
        
        if not extracted_data.get("event_code") and not extracted_data.get("wedding_name"):
            return {"match_found": False, "matched_event": None, "confidence": 0.0, "reasoning": "No identifying information provided"}
        
        system_prompt = """You are Nayna, an AI wedding assistant. Verify if the user's provided information matches any available wedding events.

MATCHING CRITERIA:
1. Exact event code match (case insensitive)
2. Wedding name similarity (fuzzy matching)
3. City name similarity
4. Couple name similarity
5. Combination of multiple factors

Respond with JSON:
{
    "match_found": true/false,
    "matched_event_index": 0-based index if found,
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation of match/no-match",
    "alternative_matches": [list of other possible matches with scores]
}"""

        user_prompt = f"""User provided data:
{json.dumps(extracted_data, indent=2)}

Available events:
{json.dumps(available_events, indent=2)}

Find the best match for the user's information."""

        try:
            response = self.oai.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=400,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            
            # Get the matched event
            if result.get("match_found") and result.get("matched_event_index") is not None:
                idx = result["matched_event_index"]
                if 0 <= idx < len(available_events):
                    result["matched_event"] = available_events[idx]
                else:
                    result["match_found"] = False
                    result["reasoning"] = "Invalid event index"
            
            return result
        except Exception as e:
            return {
                "match_found": False,
                "matched_event": None,
                "confidence": 0.0,
                "reasoning": f"Error in verification: {str(e)}"
            }
    
    def llm_generate_onboarding_response(self, session: Dict[str, Any], analysis: Dict[str, Any], verification: Optional[Dict[str, Any]] = None) -> str:
        """Use LLM to generate contextual onboarding responses"""
        
        system_prompt = """You are Nayna, a playful Gen-Z Hinglish wedding helpdesk cousin helping users get onboarded.

PERSONALITY:
- Fun, enthusiastic, supportive
- Mix of English and Hindi (Hinglish)
- Use emojis appropriately (1-2 max)
- Be encouraging and helpful
- Keep responses conversational and engaging

RESPONSE GUIDELINES:
- Acknowledge what the user said
- Guide them to the next step
- Be encouraging if they're on the right track
- Be helpful if they need clarification
- Celebrate when they complete steps
- Use casual, friendly language"""

        current_state = session.get("state", "INIT")
        session_data = session.get("data", {})
        
        user_prompt = f"""Current onboarding state: {current_state}
Session data: {json.dumps(session_data, indent=2)}
Analysis result: {json.dumps(analysis, indent=2)}
Verification result: {json.dumps(verification or {}, indent=2)}

Generate an appropriate response for this onboarding step. Consider:
1. What the user just said
2. What step they're on
3. Whether they need help or clarification
4. Whether they completed a step successfully
5. What they need to do next"""

        try:
            response = self.oai.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return "Hey! I'm having a small technical issue right now 😅 Can you try again?"
    
    async def process_onboarding_message(self, wa_id: str, user_msg: str, available_events: list) -> Dict[str, Any]:
        """Main onboarding processing function"""
        
        # Load session
        session = self.load_onb_sess(wa_id)
        
        # Add user message to conversation history
        session["conversation_history"].append({
            "timestamp": self.now_iso(),
            "role": "user",
            "message": user_msg
        })
        
        # Analyze the message
        analysis = self.llm_analyze_onboarding_step(user_msg, session, available_events)
        
        # Update session based on analysis
        next_state = analysis.get("next_state", session["state"])
        extracted_data = analysis.get("extracted_data", {})
        
        # Merge extracted data into session data
        session["data"].update(extracted_data)
        session["state"] = next_state
        
        # Handle verification if needed
        verification = None
        if next_state == "VERIFYING" and extracted_data:
            verification = self.llm_verify_event_match(extracted_data, available_events)
            
            if verification.get("match_found"):
                session["state"] = "COLLECTING_DETAILS"
                session["data"]["verified_event"] = verification["matched_event"]
            else:
                session["state"] = "INIT"
                session["data"]["verification_attempts"] = session["data"].get("verification_attempts", 0) + 1
        
        # Generate response
        response = self.llm_generate_onboarding_response(session, analysis, verification)
        
        # Add response to conversation history
        session["conversation_history"].append({
            "timestamp": self.now_iso(),
            "role": "assistant",
            "message": response
        })
        
        # Save session
        self.save_onb_sess(wa_id, session)
        
        # Determine if onboarding is complete
        is_complete = (next_state == "COMPLETED" or 
                      (next_state == "COLLECTING_DETAILS" and 
                       session["data"].get("user_name") and 
                       session["data"].get("relation")))
        
        return {
            "onboarded": is_complete,
            "reply": response,
            "forward_to_query": is_complete,
            "session_state": next_state,
            "extracted_data": extracted_data,
            "verification": verification
        }
