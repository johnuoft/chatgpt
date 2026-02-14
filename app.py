"""
Fleetwork Backend v0.5.1 — AI-Managed Marketing Agency
========================================================
Full autonomous AI PM with:
- 3-tier decision framework (handle / inform / escalate)
- Proactive daily standups + deadline management
- Two-way Slack bot + in-app chat
- Email via Resend, Slack, in-app notifications
- Campaign health scores + analytics
- Talent leaderboard + performance tracking
- Auto-generated weekly client reports
- Webhook events for frontend real-time updates

Fix: renamed 'metadata' to 'event_metadata' (SQLAlchemy reserved word)
"""

import os
import json
import re
import asyncio
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Float, DateTime,
    Boolean, ForeignKey, JSON, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
import httpx

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./growthteam.db")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"
FROM_EMAIL = os.getenv("FROM_EMAIL", "onboarding@resend.dev")
APP_URL = os.getenv("APP_URL", "https://fleetwork.studio")

connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ══════════════════════════════════════════════
# DATABASE MODELS
# ══════════════════════════════════════════════

class Talent(Base):
    __tablename__ = "talents"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    email = Column(String(200), unique=True, nullable=False)
    phone = Column(String(50), nullable=True)
    primary_role = Column(String(100), nullable=False)
    channels = Column(JSON, default=list)
    skills = Column(JSON, default=list)
    portfolio_url = Column(String(500), nullable=True)
    bio = Column(Text, nullable=True)
    hourly_rate = Column(Float, nullable=True)
    per_deliverable_rate = Column(Float, nullable=True)
    rate_notes = Column(String(500), nullable=True)
    availability_hours_per_week = Column(Integer, default=20)
    timezone = Column(String(50), default="EST")
    avg_rating = Column(Float, default=0.0)
    total_tasks_completed = Column(Integer, default=0)
    on_time_delivery_rate = Column(Float, default=1.0)
    total_revisions_requested = Column(Integer, default=0)
    response_time_hours_avg = Column(Float, default=0.0)
    status = Column(String(20), default="active")
    slack_user_id = Column(String(50), nullable=True)
    preferred_contact = Column(String(50), default="email")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    assignments = relationship("TaskAssignment", back_populates="talent")


class Client(Base):
    __tablename__ = "clients"
    id = Column(Integer, primary_key=True, index=True)
    business_name = Column(String(200), nullable=False)
    website_url = Column(String(500), nullable=False)
    contact_name = Column(String(200), nullable=False)
    contact_email = Column(String(200), nullable=False)
    slack_channel_id = Column(String(50), nullable=True)
    slack_user_id = Column(String(50), nullable=True)
    business_analysis = Column(JSON, nullable=True)
    industry = Column(String(100), nullable=True)
    target_audience = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    campaigns = relationship("Campaign", back_populates="client")


class Campaign(Base):
    __tablename__ = "campaigns"
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False)
    name = Column(String(200), nullable=False)
    strategy = Column(JSON, nullable=False)
    status = Column(String(20), default="draft")
    monthly_budget = Column(Float, nullable=True)
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    client = relationship("Client", back_populates="campaigns")
    tasks = relationship("Task", back_populates="campaign")


class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    campaign_id = Column(Integer, ForeignKey("campaigns.id"), nullable=False)
    title = Column(String(300), nullable=False)
    description = Column(Text, nullable=False)
    channel = Column(String(50), nullable=True)
    required_role = Column(String(100), nullable=False)
    required_skills = Column(JSON, default=list)
    due_date = Column(DateTime, nullable=True)
    publish_date = Column(DateTime, nullable=True)
    recurrence = Column(String(50), nullable=True)
    status = Column(String(30), default="pending")
    priority = Column(Integer, default=2)
    deliverable_url = Column(String(500), nullable=True)
    ai_review_notes = Column(Text, nullable=True)
    ai_review_score = Column(Float, nullable=True)
    client_feedback = Column(Text, nullable=True)
    revision_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    campaign = relationship("Campaign", back_populates="tasks")
    assignment = relationship("TaskAssignment", back_populates="task", uselist=False)


class TaskAssignment(Base):
    __tablename__ = "task_assignments"
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    talent_id = Column(Integer, ForeignKey("talents.id"), nullable=False)
    assigned_at = Column(DateTime, default=datetime.utcnow)
    accepted = Column(Boolean, nullable=True)
    responded_at = Column(DateTime, nullable=True)
    payment_amount = Column(Float, nullable=True)
    payment_status = Column(String(20), default="pending")
    task = relationship("Task", back_populates="assignment")
    talent = relationship("Talent", back_populates="assignments")


class MessageLog(Base):
    __tablename__ = "message_logs"
    id = Column(Integer, primary_key=True, index=True)
    direction = Column(String(20), nullable=False)
    recipient_type = Column(String(20), nullable=False)
    recipient_id = Column(Integer, nullable=False)
    channel = Column(String(20), nullable=False)
    subject = Column(String(300), nullable=True)
    body = Column(Text, nullable=False)
    related_task_id = Column(Integer, nullable=True)
    related_campaign_id = Column(Integer, nullable=True)
    status = Column(String(20), default="sent")
    sent_at = Column(DateTime, default=datetime.utcnow)


class Notification(Base):
    __tablename__ = "notifications"
    id = Column(Integer, primary_key=True, index=True)
    user_type = Column(String(20), nullable=False)
    user_id = Column(Integer, nullable=False)
    title = Column(String(300), nullable=False)
    message = Column(Text, nullable=False)
    notification_type = Column(String(50), nullable=False)
    related_task_id = Column(Integer, nullable=True)
    related_campaign_id = Column(Integer, nullable=True)
    action_url = Column(String(500), nullable=True)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ActivityEvent(Base):
    """Tracks all activity for the campaign timeline/feed."""
    __tablename__ = "activity_events"
    id = Column(Integer, primary_key=True, index=True)
    campaign_id = Column(Integer, nullable=True)
    task_id = Column(Integer, nullable=True)
    talent_id = Column(Integer, nullable=True)
    client_id = Column(Integer, nullable=True)
    event_type = Column(String(50), nullable=False)
    description = Column(Text, nullable=False)
    event_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


# ══════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════

app = FastAPI(title="Fleetwork API", description="AI-managed marketing agency", version="0.5.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ══════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ══════════════════════════════════════════════

class TalentCreate(BaseModel):
    name: str
    email: str
    primary_role: str
    channels: list[str] = []
    skills: list[str] = []
    portfolio_url: Optional[str] = None
    bio: Optional[str] = None
    hourly_rate: Optional[float] = None
    per_deliverable_rate: Optional[float] = None
    rate_notes: Optional[str] = None
    availability_hours_per_week: int = 20
    timezone: str = "EST"
    preferred_contact: str = "email"

class TalentBulkImport(BaseModel):
    talents: list[TalentCreate]

class ClientCreate(BaseModel):
    business_name: str
    website_url: str
    contact_name: str
    contact_email: str
    slack_channel_id: Optional[str] = None

class StrategyRequest(BaseModel):
    url: str
    goals: Optional[str] = None
    budget_range: Optional[str] = None

class TaskSubmission(BaseModel):
    deliverable_url: str
    notes: Optional[str] = None

class ApprovalDecision(BaseModel):
    approved: bool
    feedback: Optional[str] = None

class ChatMessage(BaseModel):
    message: str
    user_type: str = "client"
    user_id: Optional[int] = None


# ══════════════════════════════════════════════
# COMMUNICATION SYSTEM
# ══════════════════════════════════════════════

async def send_email(to, subject, html_body, text_body=None):
    if not RESEND_API_KEY:
        print(f"[EMAIL SKIP] {to}: {subject}")
        return {"status": "skipped"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post("https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
                json={"from": f"Fleetwork <{FROM_EMAIL}>", "to": [to], "subject": subject, "html": html_body, "text": text_body or ""})
            result = r.json()
            return {"status": "sent", "id": result.get("id")} if r.status_code < 400 else {"status": "failed", "error": result}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


async def send_slack(channel, text, blocks=None):
    if not channel:
        return {"status": "skipped"}
    if channel.startswith("https://"):
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                await c.post(channel, json={"text": text, **({"blocks": blocks} if blocks else {})})
                return {"status": "sent"}
        except:
            return {"status": "failed"}
    if SLACK_BOT_TOKEN:
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.post("https://slack.com/api/chat.postMessage",
                    headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                    json={"channel": channel, "text": text, **({"blocks": blocks} if blocks else {})})
                return {"status": "sent"} if r.json().get("ok") else {"status": "failed"}
        except:
            return {"status": "failed"}
    return {"status": "skipped"}


def create_notif(db, user_type, user_id, title, message, ntype, task_id=None, campaign_id=None, url=None):
    n = Notification(user_type=user_type, user_id=user_id, title=title, message=message, notification_type=ntype, related_task_id=task_id, related_campaign_id=campaign_id, action_url=url)
    db.add(n); db.commit(); return n


def log_msg(db, direction, rtype, rid, channel, body, subject=None, task_id=None, campaign_id=None, status="sent"):
    m = MessageLog(direction=direction, recipient_type=rtype, recipient_id=rid, channel=channel, subject=subject, body=body, related_task_id=task_id, related_campaign_id=campaign_id, status=status)
    db.add(m); db.commit(); return m


def log_activity(db, event_type, description, campaign_id=None, task_id=None, talent_id=None, client_id=None, event_metadata=None):
    e = ActivityEvent(campaign_id=campaign_id, task_id=task_id, talent_id=talent_id, client_id=client_id, event_type=event_type, description=description, event_metadata=event_metadata)
    db.add(e); db.commit(); return e


async def notify_talent(db, talent, subject, html, text, ntitle, nmsg, ntype, task_id=None, campaign_id=None, url=None, slack_text=None):
    create_notif(db, "talent", talent.id, ntitle, nmsg, ntype, task_id, campaign_id, url)
    r = await send_email(talent.email, subject, html, text)
    log_msg(db, "outbound_talent", "talent", talent.id, "email", text, subject, task_id, campaign_id, r.get("status"))
    if talent.slack_user_id:
        sr = await send_slack(talent.slack_user_id, slack_text or text)
        log_msg(db, "outbound_talent", "talent", talent.id, "slack", slack_text or text, None, task_id, campaign_id, sr.get("status"))


async def notify_client(db, client, subject, html, text, ntitle, nmsg, ntype, task_id=None, campaign_id=None, url=None, slack_text=None):
    create_notif(db, "client", client.id, ntitle, nmsg, ntype, task_id, campaign_id, url)
    if client.contact_email:
        r = await send_email(client.contact_email, subject, html, text)
        log_msg(db, "outbound_client", "client", client.id, "email", text, subject, task_id, campaign_id, r.get("status"))
    if client.slack_channel_id:
        sr = await send_slack(client.slack_channel_id, slack_text or text)
        log_msg(db, "outbound_client", "client", client.id, "slack", slack_text or text, None, task_id, campaign_id, sr.get("status"))


# ══════════════════════════════════════════════
# EMAIL TEMPLATES
# ══════════════════════════════════════════════

def _wrap_email(content):
    return f"<div style='font-family:Inter,sans-serif;max-width:600px;margin:0 auto;background:#09090b;color:#fafafa;padding:32px;border-radius:12px;'><div style='text-align:center;margin-bottom:24px;'><span style='font-size:24px;font-weight:700;color:#f97316;'>⚡ Fleetwork</span></div>{content}</div>"

def brief_email_html(name, title, brief, due, url):
    return _wrap_email(f"<h2 style='color:#fafafa;'>New Task Assignment</h2><p style='color:#a3a3a3;'>Hey {name}, you've been matched!</p><div style='background:#111113;border:1px solid #1c1c1e;border-radius:8px;padding:20px;margin:20px 0;'><h3 style='color:#f97316;margin-top:0;'>{title}</h3><p style='color:#d4d4d4;white-space:pre-wrap;'>{brief}</p></div><p style='color:#a3a3a3;'>📅 Due: <strong style='color:#fafafa;'>{due}</strong></p><div style='text-align:center;margin:24px 0;'><a href='{url}' style='background:#f97316;color:#fafafa;text-decoration:none;padding:12px 32px;border-radius:8px;font-weight:600;'>View Task →</a></div>")

def revision_email_html(name, title, feedback, url):
    return _wrap_email(f"<h2 style='color:#fafafa;'>Revision Requested</h2><p style='color:#a3a3a3;'>Hey {name}, feedback on your submission:</p><div style='background:#111113;border:1px solid #1c1c1e;border-radius:8px;padding:20px;margin:20px 0;'><h3 style='color:#fafafa;margin-top:0;'>{title}</h3><div style='background:#1a1a1d;border-left:3px solid #f97316;padding:12px 16px;border-radius:4px;'><p style='color:#d4d4d4;white-space:pre-wrap;margin:0;'>{feedback}</p></div></div><div style='text-align:center;margin:24px 0;'><a href='{url}' style='background:#f97316;color:#fafafa;text-decoration:none;padding:12px 32px;border-radius:8px;font-weight:600;'>Resubmit →</a></div>")


# ══════════════════════════════════════════════
# AI ENGINE
# ══════════════════════════════════════════════

async def call_claude(system, user_msg, max_tokens=4000):
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post("https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json", "x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01"},
            json={"model": CLAUDE_MODEL, "max_tokens": max_tokens, "system": system, "messages": [{"role": "user", "content": user_msg}]})
        data = r.json()
        if "error" in data:
            raise HTTPException(status_code=500, detail=data["error"]["message"])
        return "".join(b["text"] for b in data.get("content", []) if b.get("type") == "text")


def _parse_json(text):
    return json.loads(text.replace("```json", "").replace("```", "").strip())


async def ai_analyze_business(url, goals=None):
    system = """You are Fleetwork's AI strategist. Analyze the business URL and generate a comprehensive marketing strategy. Respond ONLY with valid JSON:
{"business_name":"string","industry":"string","business_summary":"string","target_audience":"string","competitive_landscape":"string","strategy_overview":"string","objectives":["string"],"kpis":[{"metric":"string","target":"string","timeframe":"string"}],"recommended_channels":["string"],"team_roles":[{"title":"string","type":"string","channel":"string","description":"string","skills":["string"],"estimated_rate":"string"}],"content_calendar":[{"day":"string","title":"string","channel":"string","format":"string","frequency":"string"}],"estimated_monthly_budget":"string","expected_timeline":"string"}"""
    msg = f"Analyze: {url}" + (f"\nGoals: {goals}" if goals else "")
    return _parse_json(await call_claude(system, msg))


async def ai_break_down_strategy(strategy, budget=None):
    system = """You are Fleetwork's AI PM. Break strategy into weekly tasks for 4 weeks. Respond ONLY with JSON:
{"tasks":[{"title":"string","description":"string","channel":"string","required_role":"string","required_skills":["string"],"week":1,"priority":1,"estimated_hours":2,"deliverable_format":"string","recurrence":"one-time"}]}
Rules: 3-6 tasks/week, 12-24 total. Week 1=foundational. Week 4=optimization. Specific titles. Detailed descriptions."""
    msg = f"Strategy: {json.dumps(strategy)}" + (f"\nBudget: {budget}" if budget else "")
    return _parse_json(await call_claude(system, msg, 4000))["tasks"]


async def ai_match_talent(task, available):
    system = 'Rank top 3 talent matches. Respond ONLY with JSON: {"matches":[{"talent_id":int,"score":float,"reason":"string"}]}'
    return _parse_json(await call_claude(system, f"Task: {json.dumps(task)}\nTalent: {json.dumps(available)}", 1000))["matches"]


async def ai_generate_brief(task, talent, client):
    system = "Write a clear task brief for a freelancer. Include: what to create, brand context, tone, specs, deadline. No JSON."
    return await call_claude(system, f"Task: {json.dumps(task)}\nTalent: {talent['name']} ({talent['primary_role']})\nClient: {client['business_name']}\nBrand: {client.get('business_analysis',{}).get('business_summary','N/A')}\nAudience: {client.get('target_audience','N/A')}", 1500)


async def ai_review_deliverable(task, url, client_ctx):
    system = 'Review the deliverable. Respond ONLY with JSON: {"quality_score":float,"passes_review":boolean,"feedback_for_talent":"string","summary_for_client":"string","recommendation":"approve"|"request_revision"}'
    return _parse_json(await call_claude(system, f"Task: {json.dumps(task)}\nSubmission: {url}\nClient: {json.dumps(client_ctx)}", 1000))


# ══════════════════════════════════════════════
# AUTONOMOUS AI PM — Decision Engine
# ══════════════════════════════════════════════

AI_PM_SYSTEM = """You are Fleetwork's autonomous AI Project Manager.

AUTHORITY LEVELS:

LEVEL 1 — Handle yourself (NEVER bother the client):
- Answer talent questions about brand, tone, audience, specs (use business_analysis)
- Answer questions about deadlines, formats, requirements
- Send deadline reminders
- Acknowledge acceptances and submissions
- Provide status updates
- Reschedule tasks by 1-3 days if talent asks
- Give feedback on resubmissions using original brief context
- Resolve clarifications using available data

LEVEL 2 — Handle but inform client (brief update):
- Talent declined (auto-reassign, notify client)
- Deadline extended 3+ days
- Task resubmitted after revision
- Weekly progress summary

LEVEL 3 — Escalate to client (ask for input):
- Budget decisions
- Strategy or channel changes
- Creative direction conflicts
- Quality concerns (3+ revisions on same task)
- Questions you cannot answer from available data
- Scope changes

RESPONSE FORMAT — JSON only:
{
    "response": "message to send back (use Slack markdown)",
    "actions": [action objects],
    "escalation": null or {"to":"client","client_id":int,"message":"string","urgency":"low|medium|high"},
    "internal_notes": "your reasoning"
}

ACTIONS:
- {"update_task_status": {"task_id": int, "new_status": "string"}}
- {"update_task_due_date": {"task_id": int, "new_due_date": "YYYY-MM-DD"}}
- {"send_message_to_talent": {"talent_id": int, "message": "string"}}
- {"send_message_to_client": {"client_id": int, "message": "string"}}
- {"create_task": {"title":"string","description":"string","channel":"string","required_role":"string","campaign_id":int}}
- {"reassign_task": {"task_id": int, "reason": "string"}}

Keep responses short for Slack (<300 words). Be warm but professional. Use emojis sparingly."""


async def ai_pm_respond(message, context, sender_type="unknown"):
    system = AI_PM_SYSTEM + f"\n\nCONTEXT:\n{json.dumps(context, indent=2, default=str)}\n\nMESSAGE FROM: {sender_type}"
    text = await call_claude(system, message, 2000)
    try:
        return _parse_json(text)
    except:
        return {"response": text, "actions": [], "escalation": None, "internal_notes": "parse failed"}


async def execute_actions(actions, db):
    results = []
    for action in actions:
        try:
            if "update_task_status" in action:
                d = action["update_task_status"]
                t = db.query(Task).filter(Task.id == d["task_id"]).first()
                if t:
                    t.status = d["new_status"]; db.commit()
                    results.append(f"Task #{d['task_id']} → {d['new_status']}")
            elif "update_task_due_date" in action:
                d = action["update_task_due_date"]
                t = db.query(Task).filter(Task.id == d["task_id"]).first()
                if t:
                    t.due_date = datetime.strptime(d["new_due_date"], "%Y-%m-%d"); db.commit()
                    results.append(f"Task #{d['task_id']} due → {d['new_due_date']}")
            elif "send_message_to_talent" in action:
                d = action["send_message_to_talent"]
                t = db.query(Talent).filter(Talent.id == d["talent_id"]).first()
                if t:
                    if t.slack_user_id:
                        await send_slack(t.slack_user_id, d["message"])
                    else:
                        await send_email(t.email, "Message from Fleetwork", _wrap_email(f"<p>{d['message']}</p>"), d["message"])
                    results.append(f"Messaged {t.name}")
            elif "create_task" in action:
                d = action["create_task"]
                task = Task(campaign_id=d["campaign_id"], title=d["title"], description=d["description"], channel=d.get("channel"), required_role=d["required_role"], status="pending", due_date=datetime.utcnow() + timedelta(days=7))
                db.add(task); db.commit()
                results.append(f"Created: {d['title']}")
        except Exception as e:
            results.append(f"Failed: {e}")
    return results


async def handle_escalation(esc, db):
    if not esc:
        return
    client = db.query(Client).filter(Client.id == esc.get("client_id")).first()
    if not client:
        return
    emoji = {"low": "ℹ️", "medium": "⚠️", "high": "🚨"}.get(esc.get("urgency", "low"), "ℹ️")
    msg = f"{emoji} *Fleetwork needs your input*\n\n{esc['message']}\n\n_Reply here to respond._"
    if client.slack_channel_id:
        await send_slack(client.slack_channel_id, msg)
    create_notif(db, "client", client.id, f"{emoji} Decision needed", esc["message"], "escalation")
    log_activity(db, "escalation", esc["message"], client_id=client.id)


def get_context(db, slack_user_id=None, client_id=None, user_type=None, user_id=None):
    """Build database context for the AI PM."""
    ctx = {"campaigns": [], "user": None}

    client = None
    if slack_user_id:
        client = db.query(Client).filter(Client.slack_user_id == slack_user_id).first()
        if not client:
            talent = db.query(Talent).filter(Talent.slack_user_id == slack_user_id).first()
            if talent:
                ctx["user"] = {"type": "talent", "id": talent.id, "name": talent.name, "role": talent.primary_role}
                for a in db.query(TaskAssignment).filter(TaskAssignment.talent_id == talent.id).all():
                    t = db.query(Task).filter(Task.id == a.task_id).first()
                    if t:
                        c = db.query(Campaign).filter(Campaign.id == t.campaign_id).first()
                        cl = db.query(Client).filter(Client.id == c.client_id).first() if c else None
                        ctx["campaigns"].append({"task_id": t.id, "title": t.title, "status": t.status, "channel": t.channel, "due_date": str(t.due_date) if t.due_date else None, "campaign_id": t.campaign_id, "campaign_name": c.name if c else None, "client_name": cl.business_name if cl else None, "client_id": cl.id if cl else None, "brand_context": cl.business_analysis.get("business_summary", "") if cl and cl.business_analysis else "", "target_audience": cl.target_audience if cl else ""})
                return ctx

    if client_id:
        client = db.query(Client).filter(Client.id == client_id).first()
    if user_type == "talent" and user_id:
        talent = db.query(Talent).filter(Talent.id == user_id).first()
        if talent:
            ctx["user"] = {"type": "talent", "id": talent.id, "name": talent.name, "role": talent.primary_role}
            for a in db.query(TaskAssignment).filter(TaskAssignment.talent_id == talent.id).all():
                t = db.query(Task).filter(Task.id == a.task_id).first()
                if t:
                    c = db.query(Campaign).filter(Campaign.id == t.campaign_id).first()
                    cl = db.query(Client).filter(Client.id == c.client_id).first() if c else None
                    ctx["campaigns"].append({"task_id": t.id, "title": t.title, "status": t.status, "channel": t.channel, "description": t.description, "due_date": str(t.due_date) if t.due_date else None, "client_name": cl.business_name if cl else None, "client_id": cl.id if cl else None, "brand_context": cl.business_analysis.get("business_summary", "") if cl and cl.business_analysis else "", "target_audience": cl.target_audience if cl else ""})
            return ctx

    if not client:
        for c in db.query(Campaign).order_by(Campaign.created_at.desc()).limit(5).all():
            tasks = db.query(Task).filter(Task.campaign_id == c.id).all()
            ctx["campaigns"].append({"campaign_id": c.id, "name": c.name, "status": c.status, "client": c.client.business_name if c.client else "?", "tasks": [{"id": t.id, "title": t.title, "status": t.status, "channel": t.channel, "due_date": str(t.due_date) if t.due_date else None, "assigned_to": (db.query(Talent).filter(Talent.id == t.assignment.talent_id).first().name if t.assignment else "Unassigned")} for t in tasks]})
        return ctx

    ctx["user"] = {"type": "client", "id": client.id, "name": client.business_name}
    for c in db.query(Campaign).filter(Campaign.client_id == client.id).order_by(Campaign.created_at.desc()).limit(5).all():
        tasks = db.query(Task).filter(Task.campaign_id == c.id).all()
        status_sum = {}
        for t in tasks:
            status_sum[t.status] = status_sum.get(t.status, 0) + 1
        ctx["campaigns"].append({"campaign_id": c.id, "name": c.name, "status": c.status, "total_tasks": len(tasks), "status_summary": status_sum, "tasks": [{"id": t.id, "title": t.title, "status": t.status, "channel": t.channel, "due_date": str(t.due_date) if t.due_date else None, "assigned_to": (db.query(Talent).filter(Talent.id == t.assignment.talent_id).first().name if t.assignment else "Unassigned"), "priority": t.priority} for t in tasks]})
    return ctx


# ══════════════════════════════════════════════
# PROACTIVE AI — Scheduled Jobs
# ══════════════════════════════════════════════

async def daily_standup():
    db = SessionLocal()
    try:
        now = datetime.utcnow()

        # Overdue tasks
        overdue = db.query(Task).filter(Task.status.in_(["assigned", "in_progress"]), Task.due_date < now).all()
        for task in overdue:
            if not task.assignment:
                continue
            talent = db.query(Talent).filter(Talent.id == task.assignment.talent_id).first()
            if not talent:
                continue
            days = (now - task.due_date).days
            campaign = db.query(Campaign).filter(Campaign.id == task.campaign_id).first()
            client = db.query(Client).filter(Client.id == campaign.client_id).first() if campaign else None

            if days <= 2:
                msg = f"Hey {talent.name}! Quick reminder — *{task.title}* was due {days} day{'s' if days != 1 else ''} ago. How's it going?"
                if talent.slack_user_id:
                    await send_slack(talent.slack_user_id, msg)
                else:
                    await send_email(talent.email, f"Reminder: {task.title}", _wrap_email(f"<p>{msg}</p>"), msg)
            elif days <= 5:
                msg = f"Hi {talent.name}, *{task.title}* is {days} days overdue. Can you give me an ETA? If you can't finish, let me know so I can reassign."
                if talent.slack_user_id:
                    await send_slack(talent.slack_user_id, msg)
                if client:
                    await handle_escalation({"client_id": client.id, "message": f"*{task.title}* is {days} days overdue (assigned to {talent.name}). I've followed up. Want me to reassign?", "urgency": "medium"}, db)
            else:
                if client:
                    await handle_escalation({"client_id": client.id, "message": f"🚨 *{task.title}* is {days} days overdue ({talent.name}). Strongly recommend reassigning.", "urgency": "high"}, db)

        # Due tomorrow
        tomorrow = now + timedelta(days=1)
        upcoming = db.query(Task).filter(Task.status.in_(["assigned", "in_progress"]), Task.due_date >= now, Task.due_date <= tomorrow).all()
        for task in upcoming:
            if task.assignment:
                talent = db.query(Talent).filter(Talent.id == task.assignment.talent_id).first()
                if talent and talent.slack_user_id:
                    await send_slack(talent.slack_user_id, f"⏰ *{task.title}* is due tomorrow! Submit when ready at {APP_URL}/talent-dashboard/tasks/{task.id}")

        # Unresponsive (48h no acceptance)
        cutoff = now - timedelta(hours=48)
        stale = db.query(TaskAssignment).filter(TaskAssignment.accepted == None, TaskAssignment.assigned_at < cutoff).all()
        for a in stale:
            talent = db.query(Talent).filter(Talent.id == a.talent_id).first()
            task = db.query(Task).filter(Task.id == a.task_id).first()
            if talent and task:
                msg = f"Hi {talent.name}! You were assigned *{task.title}* — can you take it on? Let me know either way 🙏"
                if talent.slack_user_id:
                    await send_slack(talent.slack_user_id, msg)

        # Monday weekly summary
        if now.weekday() == 0:
            for client in db.query(Client).join(Campaign).filter(Campaign.status.in_(["draft", "active"])).distinct().all():
                parts = [f"📊 *Weekly Update — {client.business_name}*\n"]
                for c in db.query(Campaign).filter(Campaign.client_id == client.id, Campaign.status.in_(["draft", "active"])).all():
                    tasks = db.query(Task).filter(Task.campaign_id == c.id).all()
                    done = len([t for t in tasks if t.status == "completed"])
                    review = len([t for t in tasks if t.status == "client_approval"])
                    overdue_count = len([t for t in tasks if t.status in ("assigned", "in_progress") and t.due_date and t.due_date < now])
                    parts.append(f"\n*{c.name}*\n✅ {done}/{len(tasks)} completed" + (f"\n👀 {review} awaiting approval" if review else "") + (f"\n⚠️ {overdue_count} overdue" if overdue_count else ""))
                summary = "\n".join(parts)
                if client.slack_channel_id:
                    await send_slack(client.slack_channel_id, summary)
                if client.contact_email:
                    await send_email(client.contact_email, f"Weekly Update — {client.business_name}", _wrap_email(f"<pre style='color:#d4d4d4;white-space:pre-wrap;'>{summary}</pre>"), summary)

        print(f"[STANDUP] Done. Overdue:{len(overdue)} Due tomorrow:{len(upcoming)} Stale:{len(stale)}")
    except Exception as e:
        print(f"[STANDUP ERROR] {e}")
        import traceback; traceback.print_exc()
    finally:
        db.close()


async def scheduler():
    while True:
        now = datetime.utcnow()
        if now.hour == 14 and now.minute < 5:  # 9 AM EST = 14 UTC
            await daily_standup()
            await asyncio.sleep(600)
        await asyncio.sleep(300)


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(scheduler())
    print("[SCHEDULER] Started")


# ══════════════════════════════════════════════
# SLACK EVENTS (Conversational Bot)
# ══════════════════════════════════════════════

@app.post("/api/slack/events")
async def slack_events(request: Request):
    body = await request.json()
    if body.get("type") == "url_verification":
        return JSONResponse(content={"challenge": body.get("challenge")})

    event = body.get("event", {})
    if event.get("bot_id") or event.get("subtype") == "bot_message":
        return {"ok": True}

    if event.get("type") in ("message", "app_mention"):
        text = event.get("text", "")
        user_id = event.get("user", "")
        channel = event.get("channel", "")
        if event.get("type") == "app_mention":
            text = re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()
        if not text:
            return {"ok": True}

        db = SessionLocal()
        try:
            ctx = get_context(db, slack_user_id=user_id)
            sender = ctx.get("user", {}).get("type", "unknown")
            result = await ai_pm_respond(text, ctx, sender)
            resp = result.get("response", "Sorry, I couldn't process that.")
            actions = result.get("actions", [])
            if actions:
                ar = await execute_actions(actions, db)
                if ar:
                    resp += "\n\n_Actions:_\n" + "\n".join(f"• {r}" for r in ar)
            if result.get("escalation"):
                await handle_escalation(result["escalation"], db)
            await send_slack(channel, resp)
            log_msg(db, "inbound", sender, ctx.get("user", {}).get("id", 0), "slack", text)
            log_msg(db, f"outbound_{sender}", sender, ctx.get("user", {}).get("id", 0), "slack", resp)
        except Exception as e:
            print(f"[SLACK ERROR] {e}"); import traceback; traceback.print_exc()
            await send_slack(channel, "Sorry, hit an issue. Try again in a moment! 🔧")
        finally:
            db.close()
    return {"ok": True}


# ══════════════════════════════════════════════
# IN-APP CHAT
# ══════════════════════════════════════════════

@app.post("/api/chat", tags=["Chat"])
async def in_app_chat(msg: ChatMessage, db: Session = Depends(get_db)):
    ctx = get_context(db, client_id=msg.user_id if msg.user_type == "client" else None, user_type=msg.user_type, user_id=msg.user_id)
    result = await ai_pm_respond(msg.message, ctx, msg.user_type)
    actions = result.get("actions", [])
    ar = await execute_actions(actions, db) if actions else []
    if result.get("escalation"):
        await handle_escalation(result["escalation"], db)
    if msg.user_id:
        log_msg(db, "inbound", msg.user_type, msg.user_id, "in_app", msg.message)
    return {"response": result.get("response", ""), "actions_taken": ar}


# ══════════════════════════════════════════════
# TALENT MANAGEMENT
# ══════════════════════════════════════════════

@app.post("/api/talent", tags=["Talent"])
def create_talent(data: TalentCreate, db: Session = Depends(get_db)):
    t = Talent(**data.model_dump()); db.add(t); db.commit(); db.refresh(t)
    log_activity(db, "talent_joined", f"{t.name} joined as {t.primary_role}", talent_id=t.id)
    return {"id": t.id, "name": t.name, "status": "created"}

@app.post("/api/talent/bulk", tags=["Talent"])
def bulk_import(data: TalentBulkImport, db: Session = Depends(get_db)):
    created, skipped = [], []
    for t in data.talents:
        if db.query(Talent).filter(Talent.email == t.email).first():
            skipped.append(t.email); continue
        db.add(Talent(**t.model_dump())); created.append(t.email)
    db.commit()
    return {"created": len(created), "skipped": len(skipped), "skipped_emails": skipped}

@app.get("/api/talent", tags=["Talent"])
def list_talent(role: Optional[str] = None, channel: Optional[str] = None, status: str = "active", db: Session = Depends(get_db)):
    q = db.query(Talent).filter(Talent.status == status)
    if role: q = q.filter(Talent.primary_role.ilike(f"%{role}%"))
    talents = q.all()
    if channel: talents = [t for t in talents if channel.lower() in [c.lower() for c in (t.channels or [])]]
    return [{"id": t.id, "name": t.name, "primary_role": t.primary_role, "channels": t.channels, "skills": t.skills, "hourly_rate": t.hourly_rate, "per_deliverable_rate": t.per_deliverable_rate, "avg_rating": t.avg_rating, "total_tasks_completed": t.total_tasks_completed, "on_time_delivery_rate": t.on_time_delivery_rate, "status": t.status} for t in talents]

@app.get("/api/talent/{tid}", tags=["Talent"])
def get_talent(tid: int, db: Session = Depends(get_db)):
    t = db.query(Talent).filter(Talent.id == tid).first()
    if not t: raise HTTPException(404, "Not found")
    return t


# ══════════════════════════════════════════════
# TALENT LEADERBOARD
# ══════════════════════════════════════════════

@app.get("/api/talent/leaderboard", tags=["Talent"])
def talent_leaderboard(db: Session = Depends(get_db)):
    talents = db.query(Talent).filter(Talent.status == "active", Talent.total_tasks_completed > 0).all()
    ranked = []
    for t in talents:
        score = (t.total_tasks_completed * t.on_time_delivery_rate * max(t.avg_rating, 3.0)) / 10
        ranked.append({"id": t.id, "name": t.name, "role": t.primary_role, "tasks_completed": t.total_tasks_completed, "on_time_rate": round(t.on_time_delivery_rate * 100), "avg_rating": t.avg_rating, "revisions": t.total_revisions_requested, "performance_score": round(score, 2)})
    ranked.sort(key=lambda x: x["performance_score"], reverse=True)
    return ranked


# ══════════════════════════════════════════════
# CAMPAIGNS
# ══════════════════════════════════════════════

@app.post("/api/clients", tags=["Clients"])
def create_client(data: ClientCreate, db: Session = Depends(get_db)):
    c = Client(**data.model_dump()); db.add(c); db.commit(); db.refresh(c)
    return {"id": c.id, "business_name": c.business_name}

@app.post("/api/campaigns/generate", tags=["Campaigns"])
async def generate_campaign(req: StrategyRequest, bg: BackgroundTasks, db: Session = Depends(get_db)):
    strategy = await ai_analyze_business(req.url, req.goals)
    client = db.query(Client).filter(Client.website_url == req.url).first()
    if not client:
        client = Client(business_name=strategy.get("business_name", ""), website_url=req.url, contact_name="", contact_email="", business_analysis=strategy, industry=strategy.get("industry"), target_audience=strategy.get("target_audience"))
        db.add(client); db.commit(); db.refresh(client)
    campaign = Campaign(client_id=client.id, name=f"{strategy['business_name']} Growth Campaign", strategy=strategy, status="draft")
    db.add(campaign); db.commit(); db.refresh(campaign)
    log_activity(db, "campaign_created", f"Campaign '{campaign.name}' created", campaign.id, client_id=client.id)
    tasks = []
    for role in strategy.get("team_roles", []):
        t = Task(campaign_id=campaign.id, title=f"{role['title']} — {role['channel']}", description=role["description"], channel=role.get("channel"), required_role=role["title"], required_skills=role.get("skills", []), status="pending", due_date=datetime.utcnow() + timedelta(days=7))
        db.add(t); db.commit(); db.refresh(t)
        tasks.append({"id": t.id, "title": t.title, "role": role["title"]})
    bg.add_task(auto_match_all, campaign.id)
    return {"campaign_id": campaign.id, "client_id": client.id, "strategy": strategy, "tasks_created": tasks, "status": "Campaign created. Matching talent."}

@app.post("/api/campaigns/{cid}/breakdown", tags=["Campaigns"])
async def breakdown(cid: int, bg: BackgroundTasks, db: Session = Depends(get_db)):
    c = db.query(Campaign).filter(Campaign.id == cid).first()
    if not c: raise HTTPException(404, "Not found")
    tasks = await ai_break_down_strategy(c.strategy, c.strategy.get("estimated_monthly_budget"))
    db.query(Task).filter(Task.campaign_id == cid, Task.status == "pending").delete(); db.commit()
    created = []
    for t in tasks:
        w = t.get("week", 1)
        task = Task(campaign_id=cid, title=t["title"], description=t["description"], channel=t.get("channel"), required_role=t["required_role"], required_skills=t.get("required_skills", []), status="pending", priority=t.get("priority", 2), due_date=datetime.utcnow() + timedelta(weeks=w), recurrence=t.get("recurrence", "one-time"))
        db.add(task); db.commit(); db.refresh(task)
        created.append({"id": task.id, "title": task.title, "week": w, "channel": task.channel, "role": task.required_role, "due": task.due_date.isoformat()})
    bg.add_task(auto_match_all, cid)
    return {"campaign_id": cid, "total": len(created), "tasks_by_week": {f"week_{w}": [t for t in created if t["week"] == w] for w in range(1, 5)}}

@app.post("/api/campaigns/generate-full", tags=["Campaigns"])
async def generate_full(req: StrategyRequest, bg: BackgroundTasks, db: Session = Depends(get_db)):
    strategy = await ai_analyze_business(req.url, req.goals)
    client = db.query(Client).filter(Client.website_url == req.url).first()
    if not client:
        client = Client(business_name=strategy.get("business_name", ""), website_url=req.url, contact_name="", contact_email="", business_analysis=strategy, industry=strategy.get("industry"), target_audience=strategy.get("target_audience"))
        db.add(client); db.commit(); db.refresh(client)
    campaign = Campaign(client_id=client.id, name=f"{strategy['business_name']} Growth Campaign", strategy=strategy, status="draft")
    db.add(campaign); db.commit(); db.refresh(campaign)
    log_activity(db, "campaign_created", f"Campaign '{campaign.name}' created (full)", campaign.id, client_id=client.id)
    try:
        task_list = await ai_break_down_strategy(strategy, strategy.get("estimated_monthly_budget"))
    except:
        return {"campaign_id": campaign.id, "client_id": client.id, "strategy": strategy, "tasks_by_week": {}, "total_tasks": 0, "status": "Strategy OK, task breakdown failed. Retry /breakdown."}
    created = []
    for t in task_list:
        w = t.get("week", 1)
        task = Task(campaign_id=campaign.id, title=t["title"], description=t["description"], channel=t.get("channel"), required_role=t["required_role"], required_skills=t.get("required_skills", []), status="pending", priority=t.get("priority", 2), due_date=datetime.utcnow() + timedelta(weeks=w), recurrence=t.get("recurrence", "one-time"))
        db.add(task); db.commit(); db.refresh(task)
        created.append({"id": task.id, "title": task.title, "week": w, "channel": task.channel, "role": task.required_role, "due": task.due_date.isoformat()})
    bg.add_task(auto_match_all, campaign.id)
    return {"campaign_id": campaign.id, "client_id": client.id, "strategy": strategy, "tasks_by_week": {f"week_{w}": [t for t in created if t["week"] == w] for w in range(1, 5)}, "total_tasks": len(created), "status": "Full campaign generated."}


# ══════════════════════════════════════════════
# BACKGROUND: Auto-match + briefs
# ══════════════════════════════════════════════

async def auto_match_all(campaign_id):
    db = SessionLocal()
    try:
        tasks = db.query(Task).filter(Task.campaign_id == campaign_id, Task.status == "pending").all()
        campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        client = db.query(Client).filter(Client.id == campaign.client_id).first() if campaign else None
        for task in tasks:
            avail = db.query(Talent).filter(Talent.status == "active", Talent.primary_role.ilike(f"%{task.required_role.split()[0]}%")).all()
            if not avail: avail = db.query(Talent).filter(Talent.status == "active").all()
            if not avail: continue
            td = {"title": task.title, "description": task.description, "channel": task.channel, "required_role": task.required_role, "required_skills": task.required_skills}
            tl = [{"talent_id": t.id, "name": t.name, "primary_role": t.primary_role, "channels": t.channels, "skills": t.skills, "hourly_rate": t.hourly_rate, "per_deliverable_rate": t.per_deliverable_rate, "avg_rating": t.avg_rating, "total_tasks_completed": t.total_tasks_completed, "availability_hours_per_week": t.availability_hours_per_week} for t in avail[:20]]
            try:
                matches = await ai_match_talent(td, tl)
                if matches:
                    talent = db.query(Talent).filter(Talent.id == matches[0]["talent_id"]).first()
                    if not talent: continue
                    a = TaskAssignment(task_id=task.id, talent_id=talent.id, payment_amount=talent.per_deliverable_rate)
                    db.add(a); task.status = "assigned"; db.commit()
                    log_activity(db, "task_assigned", f"'{task.title}' assigned to {talent.name}", campaign_id, task.id, talent.id)
                    if client:
                        try: brief = await ai_generate_brief(td, {"name": talent.name, "primary_role": talent.primary_role}, {"business_name": client.business_name, "industry": client.industry, "business_analysis": client.business_analysis, "target_audience": client.target_audience})
                        except: brief = task.description
                        due = task.due_date.strftime("%B %d, %Y") if task.due_date else "TBD"
                        url = f"{APP_URL}/talent-dashboard/tasks/{task.id}"
                        await notify_talent(db, talent, f"New task: {task.title}", brief_email_html(talent.name, task.title, brief, due, url), f"New task: {task.title}\n\n{brief}\n\nDue: {due}", f"New Task: {task.title}", f"Matched to '{task.title}' for {client.business_name}", "task_assigned", task.id, campaign_id, url, f"📋 *New Task*\n*{task.title}*\nClient: {client.business_name}\nDue: {due}\n<{url}|View Brief>")
                        log_activity(db, "brief_sent", f"Brief sent to {talent.name} for '{task.title}'", campaign_id, task.id, talent.id)
            except Exception as e:
                print(f"Match error task {task.id}: {e}")
    finally:
        db.close()


# ══════════════════════════════════════════════
# TASK LIFECYCLE
# ══════════════════════════════════════════════

@app.get("/api/campaigns/{cid}/tasks", tags=["Tasks"])
def list_tasks(cid: int, db: Session = Depends(get_db)):
    tasks = db.query(Task).filter(Task.campaign_id == cid).all()
    return [{"id": t.id, "title": t.title, "channel": t.channel, "required_role": t.required_role, "status": t.status, "priority": t.priority, "revision_count": t.revision_count, "ai_score": t.ai_review_score, "due_date": t.due_date.isoformat() if t.due_date else None, "assigned_talent": {"id": t.assignment.talent_id, "name": db.query(Talent).filter(Talent.id == t.assignment.talent_id).first().name, "accepted": t.assignment.accepted} if t.assignment else None} for t in tasks]

@app.post("/api/tasks/{tid}/accept", tags=["Tasks"])
def accept_task(tid: int, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == tid).first()
    if not task: raise HTTPException(404, "Not found")
    if task.assignment:
        task.assignment.accepted = True
        task.assignment.responded_at = datetime.utcnow()
        task.status = "in_progress"
        db.commit()
        log_activity(db, "task_accepted", f"Task '{task.title}' accepted", task.campaign_id, tid, task.assignment.talent_id)
    return {"status": "accepted"}

@app.post("/api/tasks/{tid}/decline", tags=["Tasks"])
async def decline_task(tid: int, bg: BackgroundTasks, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == tid).first()
    if not task: raise HTTPException(404, "Not found")
    if task.assignment:
        old_talent = db.query(Talent).filter(Talent.id == task.assignment.talent_id).first()
        task.assignment.accepted = False
        task.assignment.responded_at = datetime.utcnow()
        db.delete(task.assignment)
        task.status = "pending"
        db.commit()
        log_activity(db, "task_declined", f"'{task.title}' declined by {old_talent.name if old_talent else '?'}", task.campaign_id, tid)
        bg.add_task(auto_match_all, task.campaign_id)
        campaign = db.query(Campaign).filter(Campaign.id == task.campaign_id).first()
        if campaign:
            client = db.query(Client).filter(Client.id == campaign.client_id).first()
            if client:
                await handle_escalation({"client_id": client.id, "message": f"*{task.title}* was declined by {old_talent.name if old_talent else 'a creator'}. I'm auto-reassigning to the next best match.", "urgency": "low"}, db)
    return {"status": "declined, reassigning"}

@app.post("/api/tasks/{tid}/submit", tags=["Tasks"])
async def submit(tid: int, sub: TaskSubmission, bg: BackgroundTasks, db: Session = Depends(get_db)):
    t = db.query(Task).filter(Task.id == tid).first()
    if not t: raise HTTPException(404, "Not found")
    t.deliverable_url = sub.deliverable_url; t.status = "review"; db.commit()
    log_activity(db, "deliverable_submitted", f"Deliverable submitted for '{t.title}'", t.campaign_id, tid, t.assignment.talent_id if t.assignment else None)
    bg.add_task(review_and_route, tid, sub.deliverable_url)
    return {"status": "Submitted. AI reviewing."}

async def review_and_route(tid, url):
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == tid).first()
        if not task: return
        camp = db.query(Campaign).filter(Campaign.id == task.campaign_id).first()
        client = db.query(Client).filter(Client.id == camp.client_id).first()
        td = {"title": task.title, "description": task.description, "channel": task.channel, "required_role": task.required_role}
        cc = {"business_name": client.business_name, "industry": client.industry, "target_audience": client.target_audience}
        review = await ai_review_deliverable(td, url, cc)
        task.ai_review_notes = json.dumps(review)
        task.ai_review_score = review.get("quality_score")

        if review.get("recommendation") == "approve":
            task.status = "client_approval"; db.commit()
            talent_name = "Your team"
            if task.assignment:
                tal = db.query(Talent).filter(Talent.id == task.assignment.talent_id).first()
                if tal: talent_name = tal.name
            dash = f"{APP_URL}/dashboard/approvals"
            await notify_client(db, client, f"✅ Review: {task.title}", _wrap_email(f"<h2>Deliverable Ready</h2><p>{task.title} by {talent_name}</p><p>{review.get('summary_for_client','')}</p><a href='{dash}' style='background:#f97316;color:white;padding:12px 24px;border-radius:8px;text-decoration:none;'>Review →</a>"), f"Ready: {task.title} by {talent_name}. Score: {review.get('quality_score')}/10", f"Review: {task.title}", f"{talent_name} submitted '{task.title}'. Score: {review.get('quality_score')}/10", "deliverable_approved", task.id, camp.id, dash, f"✅ *Review*\n*{task.title}* by {talent_name}\nScore: {review.get('quality_score')}/10\n<{dash}|Approve / Revise>")
            log_activity(db, "review_passed", f"AI approved '{task.title}' (score: {review.get('quality_score')})", camp.id, tid)
        else:
            task.status = "revision_requested"; task.revision_count += 1; db.commit()
            if task.assignment:
                tal = db.query(Talent).filter(Talent.id == task.assignment.talent_id).first()
                if tal:
                    tal.total_revisions_requested += 1; db.commit()
                    fb = review.get("feedback_for_talent", "Please revise.")
                    dash = f"{APP_URL}/talent-dashboard/tasks/{task.id}"
                    await notify_talent(db, tal, f"Revision: {task.title}", revision_email_html(tal.name, task.title, fb, dash), f"Revision: {task.title}\n{fb}", f"Revision: {task.title}", fb, "revision_requested", task.id, camp.id, dash, f"🔄 *Revision*\n*{task.title}*\n{fb}\n<{dash}|Resubmit>")
                    log_activity(db, "revision_requested", f"Revision #{task.revision_count} for '{task.title}'", camp.id, tid, tal.id)
                    if task.revision_count >= 3 and client:
                        await handle_escalation({"client_id": client.id, "message": f"⚠️ *{task.title}* has had {task.revision_count} revisions (assigned to {tal.name}). Consider reassigning or adjusting the brief.", "urgency": "medium"}, db)
    finally:
        db.close()

@app.post("/api/tasks/{tid}/approve", tags=["Tasks"])
async def approve_task(tid: int, decision: ApprovalDecision, db: Session = Depends(get_db)):
    t = db.query(Task).filter(Task.id == tid).first()
    if not t: raise HTTPException(404, "Not found")
    camp = db.query(Campaign).filter(Campaign.id == t.campaign_id).first()
    if decision.approved:
        t.status = "completed"; t.client_feedback = decision.feedback
        if t.assignment:
            tal = db.query(Talent).filter(Talent.id == t.assignment.talent_id).first()
            if tal:
                tal.total_tasks_completed += 1
                if t.due_date and datetime.utcnow() <= t.due_date:
                    tal.on_time_delivery_rate = ((tal.on_time_delivery_rate * (tal.total_tasks_completed - 1)) + 1.0) / tal.total_tasks_completed
                else:
                    tal.on_time_delivery_rate = ((tal.on_time_delivery_rate * (tal.total_tasks_completed - 1)) + 0.0) / tal.total_tasks_completed
                t.assignment.payment_status = "paid"
                await notify_talent(db, tal, f"✅ Approved: {t.title}", _wrap_email(f"<h2 style='color:#22c55e;'>✅ Approved!</h2><p>{t.title}</p><p style='color:#22c55e;'>💰 Payment processing</p>"), f"Approved: {t.title}. Payment coming.", f"Approved ✅ {t.title}", "Approved! Payment processing.", "deliverable_approved", tid, camp.id if camp else None, None, f"✅ *Approved!* {t.title} — 💰 Payment processing")
                log_activity(db, "approved", f"'{t.title}' approved", camp.id if camp else None, tid, tal.id)
    else:
        t.status = "revision_requested"; t.client_feedback = decision.feedback; t.revision_count += 1
        if t.assignment:
            tal = db.query(Talent).filter(Talent.id == t.assignment.talent_id).first()
            if tal:
                tal.total_revisions_requested += 1
                fb = decision.feedback or "Client requested changes."
                url = f"{APP_URL}/talent-dashboard/tasks/{tid}"
                await notify_talent(db, tal, f"Revision: {t.title}", revision_email_html(tal.name, t.title, fb, url), f"Revision: {t.title}\n{fb}", f"Revision: {t.title}", fb, "revision_requested", tid, camp.id if camp else None, url, f"🔄 *Revision* {t.title}\n{fb}")
    db.commit()
    return {"status": "approved" if decision.approved else "revision_requested"}


# ══════════════════════════════════════════════
# NOTIFICATIONS & MESSAGES
# ══════════════════════════════════════════════

@app.get("/api/notifications/{utype}/{uid}", tags=["Notifications"])
def get_notifs(utype: str, uid: int, unread: bool = False, db: Session = Depends(get_db)):
    q = db.query(Notification).filter(Notification.user_type == utype, Notification.user_id == uid)
    if unread: q = q.filter(Notification.is_read == False)
    return [{"id": n.id, "title": n.title, "message": n.message, "type": n.notification_type, "is_read": n.is_read, "url": n.action_url, "created_at": n.created_at.isoformat()} for n in q.order_by(Notification.created_at.desc()).limit(50).all()]

@app.post("/api/notifications/{nid}/read", tags=["Notifications"])
def mark_read(nid: int, db: Session = Depends(get_db)):
    n = db.query(Notification).filter(Notification.id == nid).first()
    if n: n.is_read = True; db.commit()
    return {"ok": True}

@app.post("/api/notifications/{utype}/{uid}/read-all", tags=["Notifications"])
def mark_all_read(utype: str, uid: int, db: Session = Depends(get_db)):
    db.query(Notification).filter(Notification.user_type == utype, Notification.user_id == uid, Notification.is_read == False).update({"is_read": True}); db.commit()
    return {"ok": True}

@app.get("/api/messages/{rtype}/{rid}", tags=["Messages"])
def get_msgs(rtype: str, rid: int, db: Session = Depends(get_db)):
    return [{"id": m.id, "direction": m.direction, "channel": m.channel, "subject": m.subject, "body": m.body, "status": m.status, "sent_at": m.sent_at.isoformat()} for m in db.query(MessageLog).filter(MessageLog.recipient_type == rtype, MessageLog.recipient_id == rid).order_by(MessageLog.sent_at.desc()).limit(50).all()]


# ══════════════════════════════════════════════
# CAMPAIGN HEALTH SCORE
# ══════════════════════════════════════════════

@app.get("/api/campaigns/{cid}/health", tags=["Analytics"])
def campaign_health(cid: int, db: Session = Depends(get_db)):
    campaign = db.query(Campaign).filter(Campaign.id == cid).first()
    if not campaign: raise HTTPException(404, "Not found")
    tasks = db.query(Task).filter(Task.campaign_id == cid).all()
    if not tasks: return {"score": 0, "grade": "N/A", "breakdown": {}}

    now = datetime.utcnow()
    total = len(tasks)
    completed = len([t for t in tasks if t.status == "completed"])
    overdue = len([t for t in tasks if t.status in ("assigned", "in_progress", "pending") and t.due_date and t.due_date < now])
    avg_revisions = sum(t.revision_count for t in tasks) / total
    avg_score = sum(t.ai_review_score or 0 for t in tasks if t.ai_review_score) / max(len([t for t in tasks if t.ai_review_score]), 1)

    completion_score = (completed / total) * 40
    timeline_score = max(0, (1 - overdue / total)) * 30
    quality_score = min(avg_score / 10, 1) * 20 if avg_score else 10
    revision_score = max(0, (1 - avg_revisions / 3)) * 10

    health = round(completion_score + timeline_score + quality_score + revision_score)
    grade = "A" if health >= 85 else "B" if health >= 70 else "C" if health >= 55 else "D" if health >= 40 else "F"

    return {
        "score": health, "grade": grade,
        "breakdown": {
            "completion": {"value": f"{completed}/{total}", "score": round(completion_score)},
            "timeline": {"overdue": overdue, "score": round(timeline_score)},
            "quality": {"avg_ai_score": round(avg_score, 1), "score": round(quality_score)},
            "revisions": {"avg_per_task": round(avg_revisions, 1), "score": round(revision_score)},
        },
        "status_breakdown": {s: len([t for t in tasks if t.status == s]) for s in set(t.status for t in tasks)},
        "at_risk_tasks": [{"id": t.id, "title": t.title, "days_overdue": (now - t.due_date).days} for t in tasks if t.status in ("assigned", "in_progress") and t.due_date and t.due_date < now],
    }


# ══════════════════════════════════════════════
# ACTIVITY FEED
# ══════════════════════════════════════════════

@app.get("/api/campaigns/{cid}/activity", tags=["Analytics"])
def campaign_activity(cid: int, limit: int = 30, db: Session = Depends(get_db)):
    events = db.query(ActivityEvent).filter(ActivityEvent.campaign_id == cid).order_by(ActivityEvent.created_at.desc()).limit(limit).all()
    return [{"id": e.id, "type": e.event_type, "description": e.description, "created_at": e.created_at.isoformat(), "metadata": e.event_metadata} for e in events]

@app.get("/api/activity/recent", tags=["Analytics"])
def recent_activity(limit: int = 30, db: Session = Depends(get_db)):
    events = db.query(ActivityEvent).order_by(ActivityEvent.created_at.desc()).limit(limit).all()
    return [{"id": e.id, "type": e.event_type, "description": e.description, "campaign_id": e.campaign_id, "created_at": e.created_at.isoformat()} for e in events]


# ══════════════════════════════════════════════
# ANALYTICS DASHBOARD
# ══════════════════════════════════════════════

@app.get("/api/analytics/overview", tags=["Analytics"])
def analytics_overview(db: Session = Depends(get_db)):
    now = datetime.utcnow()
    total_campaigns = db.query(Campaign).count()
    active_campaigns = db.query(Campaign).filter(Campaign.status.in_(["draft", "active"])).count()
    total_tasks = db.query(Task).count()
    completed_tasks = db.query(Task).filter(Task.status == "completed").count()
    active_talent = db.query(Talent).filter(Talent.status == "active").count()
    total_clients = db.query(Client).count()
    week_ago = now - timedelta(days=7)
    tasks_this_week = db.query(Task).filter(Task.status == "completed", Task.updated_at >= week_ago).count()
    scored = db.query(func.avg(Task.ai_review_score)).filter(Task.ai_review_score != None).scalar()

    return {
        "campaigns": {"total": total_campaigns, "active": active_campaigns},
        "tasks": {"total": total_tasks, "completed": completed_tasks, "completion_rate": round(completed_tasks / total_tasks * 100) if total_tasks else 0, "completed_this_week": tasks_this_week},
        "talent": {"active": active_talent},
        "clients": {"total": total_clients},
        "quality": {"avg_ai_score": round(float(scored), 1) if scored else 0},
    }


# ══════════════════════════════════════════════
# SLACK INTERACTIONS (Buttons)
# ══════════════════════════════════════════════

@app.post("/api/slack/interactions", tags=["Slack"])
async def slack_interaction(payload: dict):
    a = payload.get("actions", [{}])[0]
    aid = a.get("action_id", ""); tid = int(a.get("value", 0))
    if "approve" in aid:
        db = SessionLocal()
        t = db.query(Task).filter(Task.id == tid).first()
        if t: t.status = "completed"; db.commit()
        db.close()
        return {"text": "✅ Approved!"}
    elif "revision" in aid:
        db = SessionLocal()
        t = db.query(Task).filter(Task.id == tid).first()
        if t: t.status = "revision_requested"; db.commit()
        db.close()
        return {"text": "🔄 Revision requested."}
    return {"ok": True}


# ══════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════

@app.get("/api/campaigns/{cid}/dashboard", tags=["Dashboard"])
def dashboard(cid: int, db: Session = Depends(get_db)):
    c = db.query(Campaign).filter(Campaign.id == cid).first()
    if not c: raise HTTPException(404, "Not found")
    tasks = db.query(Task).filter(Task.campaign_id == cid).all()
    sc = {}
    for t in tasks: sc[t.status] = sc.get(t.status, 0) + 1
    return {
        "campaign_name": c.name, "status": c.status, "task_summary": sc, "total_tasks": len(tasks),
        "pending_approvals": [{"task_id": t.id, "title": t.title, "channel": t.channel, "url": t.deliverable_url, "ai_review": json.loads(t.ai_review_notes) if t.ai_review_notes else None, "ai_score": t.ai_review_score} for t in tasks if t.status == "client_approval"],
        "team": [{"task": t.title, "talent": (db.query(Talent).filter(Talent.id == t.assignment.talent_id).first().name if t.assignment else "Matching..."), "status": t.status, "due": t.due_date.isoformat() if t.due_date else None} for t in tasks],
    }


# ══════════════════════════════════════════════
# ADMIN
# ══════════════════════════════════════════════

@app.post("/api/admin/run-standup", tags=["Admin"])
async def manual_standup():
    await daily_standup()
    return {"status": "done"}

@app.get("/health")
def health():
    return {"status": "ok", "service": "fleetwork", "version": "0.5.1"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
