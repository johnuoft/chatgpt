"""
Fleetwork Backend — AI-Managed Marketing Agency Platform
==========================================================
Core systems:
1. Talent Database & Matching Engine
2. AI Project Manager (strategy → tasks → talent assignment → communication)
3. Slack Approval Bot (client only sees approve/reject decisions)
4. Campaign Orchestrator (scheduling, status tracking, deliverables)

Stack: FastAPI + SQLAlchemy + PostgreSQL
AI: Anthropic Claude API for all intelligence
Comms: Slack SDK for client approvals, email for talent coordination
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Float, DateTime,
    Boolean, ForeignKey, Enum as SQLEnum, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

# ──────────────────────────────────────────────
# Database Models
# ──────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./growthteam.db")
connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class TalentStatus(str, Enum):
    active = "active"
    inactive = "inactive"
    busy = "busy"


class TaskStatus(str, Enum):
    pending = "pending"
    assigned = "assigned"
    in_progress = "in_progress"
    review = "review"
    client_approval = "client_approval"
    approved = "approved"
    revision_requested = "revision_requested"
    completed = "completed"


class CampaignStatus(str, Enum):
    draft = "draft"
    active = "active"
    paused = "paused"
    completed = "completed"


# ── Talent ──

class Talent(Base):
    """A real human marketing specialist in the talent pool."""
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

    status = Column(String(20), default="active")
    slack_user_id = Column(String(50), nullable=True)
    preferred_contact = Column(String(50), default="email")

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    assignments = relationship("TaskAssignment", back_populates="talent")


# ── Client / Business ──

class Client(Base):
    """A business using Fleetwork."""
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


# ── Campaign ──

class Campaign(Base):
    """A marketing campaign (generated from AI strategy)."""
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


# ── Task ──

class Task(Base):
    """An individual task within a campaign."""
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
    client_feedback = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    campaign = relationship("Campaign", back_populates="tasks")
    assignment = relationship("TaskAssignment", back_populates="task", uselist=False)


# ── Task Assignment ──

class TaskAssignment(Base):
    """Links a talent to a specific task."""
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


# ── Message Log ──

class MessageLog(Base):
    """Tracks all messages sent by the AI project manager."""
    __tablename__ = "message_logs"

    id = Column(Integer, primary_key=True, index=True)
    direction = Column(String(20), nullable=False)
    recipient_type = Column(String(20), nullable=False)
    recipient_id = Column(Integer, nullable=False)
    channel = Column(String(20), nullable=False)
    subject = Column(String(300), nullable=True)
    body = Column(Text, nullable=False)
    related_task_id = Column(Integer, nullable=True)
    sent_at = Column(DateTime, default=datetime.utcnow)


# Create all tables
Base.metadata.create_all(bind=engine)


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────

app = FastAPI(
    title="Fleetwork API",
    description="AI-managed marketing agency backend",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ──────────────────────────────────────────────
# Pydantic Schemas
# ──────────────────────────────────────────────

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
    """For importing your existing list of 200+ talent."""
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


# ──────────────────────────────────────────────
# AI Engine (Claude API Integration)
# ──────────────────────────────────────────────

import httpx

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"


async def call_claude(system_prompt: str, user_message: str, max_tokens: int = 4000) -> str:
    """Call Claude API and return the text response."""
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": CLAUDE_MODEL,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_message}],
            },
        )
        data = response.json()
        if "error" in data:
            raise HTTPException(status_code=500, detail=data["error"]["message"])
        return "".join(
            block["text"] for block in data.get("content", []) if block.get("type") == "text"
        )


async def ai_analyze_business(url: str, goals: str = None) -> dict:
    """Analyze a business URL and generate full marketing strategy."""
    system = """You are Fleetwork's AI strategist. Analyze the business URL and generate a 
comprehensive marketing strategy. Respond ONLY with valid JSON matching this schema:
{
    "business_name": "string",
    "industry": "string",
    "business_summary": "string",
    "target_audience": "string",
    "competitive_landscape": "string",
    "strategy_overview": "string",
    "objectives": ["string"],
    "kpis": [{"metric": "string", "target": "string", "timeframe": "string"}],
    "recommended_channels": ["string"],
    "team_roles": [{
        "title": "string", "type": "string", "channel": "string",
        "description": "string", "skills": ["string"], "estimated_rate": "string"
    }],
    "content_calendar": [{
        "day": "string", "title": "string", "channel": "string",
        "format": "string", "frequency": "string"
    }],
    "estimated_monthly_budget": "string",
    "expected_timeline": "string"
}"""

    user_msg = f"Analyze: {url}"
    if goals:
        user_msg += f"\nClient goals: {goals}"

    text = await call_claude(system, user_msg)
    clean = text.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


async def ai_break_down_strategy(strategy: dict, budget: str = None, team_size: str = None) -> list:
    """Take a strategy and break it into specific weekly tasks for 4 weeks."""
    system = """You are Fleetwork's AI project manager. Take a marketing strategy and break it 
down into specific, actionable weekly tasks for the first 4 weeks.

Each task should be a concrete deliverable that one person can complete.

Respond ONLY with valid JSON matching this schema:
{
    "tasks": [
        {
            "title": "string (specific deliverable name)",
            "description": "string (detailed specs: what to create, dimensions, length, tone, references)",
            "channel": "string (TikTok, Instagram, YouTube, LinkedIn, SEO, Email, Paid Ads)",
            "required_role": "string (UGC Creator, Video Editor, Graphic Designer, Copywriter, Media Buyer, SEO Specialist, Social Media Manager, Email Marketer)",
            "required_skills": ["string"],
            "week": 1,
            "priority": 1,
            "estimated_hours": 2,
            "deliverable_format": "string (e.g. 'MP4 video 9:16, 30-60 seconds' or '1080x1080 PNG' or '1500 word blog post')",
            "recurrence": "one-time" or "weekly" or "biweekly"
        }
    ]
}

Rules:
- Create 3-6 tasks per week (12-24 total across 4 weeks)
- Week 1 should be foundational: brand assets, first content pieces, account setup
- Week 2-3 ramp up content production
- Week 4 includes analysis and optimization tasks
- Each task title should be specific, not vague (e.g. "3 TikTok product showcase videos (30s each)" not "Create TikTok content")
- Description should be detailed enough for a freelancer to start working immediately
- Priority: 1=critical, 2=important, 3=nice-to-have
- Match tasks to the strategy's recommended channels and team roles
- Stay within budget context if provided"""

    user_msg = f"Break this strategy into weekly tasks:\n\nStrategy: {json.dumps(strategy)}"
    if budget:
        user_msg += f"\nBudget context: {budget}"
    if team_size:
        user_msg += f"\nTeam size: {team_size}"

    text = await call_claude(system, user_msg, max_tokens=4000)
    clean = text.replace("```json", "").replace("```", "").strip()
    result = json.loads(clean)
    return result["tasks"]


async def ai_match_talent(task: dict, available_talent: list[dict]) -> list[dict]:
    """Use AI to rank and match the best talent for a specific task."""
    system = """You are Fleetwork's talent matching engine. Given a task and a list of 
available talent, rank the top 3 best matches. Consider: role match, channel expertise, 
skills overlap, availability, rating, and rate fit.

Respond ONLY with JSON: {"matches": [{"talent_id": int, "score": float, "reason": "string"}]}"""

    user_msg = f"""Task: {json.dumps(task)}

Available talent: {json.dumps(available_talent)}

Return top 3 matches ranked by fit."""

    text = await call_claude(system, user_msg, max_tokens=1000)
    clean = text.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)["matches"]


async def ai_generate_brief(task: dict, talent: dict, client: dict) -> str:
    """Generate a professional brief to send to the assigned talent."""
    system = """You are Fleetwork's AI project manager. Write a clear, professional task 
brief for a freelance marketing specialist. Include: what to create, brand context, 
tone/style guidance, deliverable specs, and deadline. Keep it concise but thorough.
Respond with just the brief text, no JSON."""

    user_msg = f"""Write a task brief for this assignment:

Task: {json.dumps(task)}
Talent: {talent['name']} ({talent['primary_role']})
Client: {client['business_name']} ({client.get('industry', 'N/A')})
Brand context: {client.get('business_analysis', {}).get('business_summary', 'N/A')}
Target audience: {client.get('target_audience', 'N/A')}"""

    return await call_claude(system, user_msg, max_tokens=1500)


async def ai_review_deliverable(task: dict, submission_url: str, client_context: dict) -> dict:
    """AI reviews a submitted deliverable before sending to client for approval."""
    system = """You are Fleetwork's AI quality reviewer. Review the submitted deliverable 
against the task requirements. Provide a quality assessment.

Respond ONLY with JSON:
{
    "quality_score": float (1-10),
    "passes_review": boolean,
    "feedback_for_talent": "string (if needs revision)",
    "summary_for_client": "string (brief description for approval)",
    "recommendation": "approve" | "request_revision"
}"""

    user_msg = f"""Review this submission:
Task: {json.dumps(task)}
Submission URL: {submission_url}
Client context: {json.dumps(client_context)}"""

    text = await call_claude(system, user_msg, max_tokens=1000)
    clean = text.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


async def ai_compose_slack_approval(task: dict, review: dict, talent_name: str) -> dict:
    """Compose a Slack message for client approval."""
    system = """You are Fleetwork's AI assistant composing a Slack message for the client.
Keep it brief and actionable. The client should be able to approve or request changes quickly.
Respond ONLY with JSON: {"text": "string", "blocks": [slack block kit elements]}"""

    user_msg = f"""Compose a Slack approval message:
Task: {task['title']}
Talent: {talent_name}
AI Review: {json.dumps(review)}

Keep it to 2-3 lines max. Include the deliverable link."""

    text = await call_claude(system, user_msg, max_tokens=800)
    clean = text.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


# ──────────────────────────────────────────────
# API Routes — Talent Management
# ──────────────────────────────────────────────

@app.post("/api/talent", tags=["Talent"])
def create_talent(talent_data: TalentCreate, db: Session = Depends(get_db)):
    """Add a single talent to the pool."""
    talent = Talent(**talent_data.model_dump())
    db.add(talent)
    db.commit()
    db.refresh(talent)
    return {"id": talent.id, "name": talent.name, "status": "created"}


@app.post("/api/talent/bulk", tags=["Talent"])
def bulk_import_talent(data: TalentBulkImport, db: Session = Depends(get_db)):
    """Bulk import talent (for your 200+ list)."""
    created = []
    skipped = []
    for t in data.talents:
        existing = db.query(Talent).filter(Talent.email == t.email).first()
        if existing:
            skipped.append(t.email)
            continue
        talent = Talent(**t.model_dump())
        db.add(talent)
        created.append(t.email)
    db.commit()
    return {
        "created": len(created),
        "skipped": len(skipped),
        "skipped_emails": skipped,
    }


@app.get("/api/talent", tags=["Talent"])
def list_talent(
    role: Optional[str] = None,
    channel: Optional[str] = None,
    status: str = "active",
    db: Session = Depends(get_db),
):
    """List talent with optional filters."""
    query = db.query(Talent).filter(Talent.status == status)
    if role:
        query = query.filter(Talent.primary_role.ilike(f"%{role}%"))
    talents = query.all()

    if channel:
        talents = [t for t in talents if channel.lower() in [c.lower() for c in (t.channels or [])]]

    return [
        {
            "id": t.id,
            "name": t.name,
            "primary_role": t.primary_role,
            "channels": t.channels,
            "skills": t.skills,
            "hourly_rate": t.hourly_rate,
            "per_deliverable_rate": t.per_deliverable_rate,
            "rate_notes": t.rate_notes,
            "avg_rating": t.avg_rating,
            "total_tasks_completed": t.total_tasks_completed,
            "availability_hours_per_week": t.availability_hours_per_week,
            "status": t.status,
        }
        for t in talents
    ]


@app.get("/api/talent/{talent_id}", tags=["Talent"])
def get_talent(talent_id: int, db: Session = Depends(get_db)):
    """Get detailed talent profile."""
    talent = db.query(Talent).filter(Talent.id == talent_id).first()
    if not talent:
        raise HTTPException(status_code=404, detail="Talent not found")
    return talent


# ──────────────────────────────────────────────
# API Routes — Client & Campaign
# ──────────────────────────────────────────────

@app.post("/api/clients", tags=["Clients"])
def create_client(client_data: ClientCreate, db: Session = Depends(get_db)):
    """Register a new client/business."""
    client = Client(**client_data.model_dump())
    db.add(client)
    db.commit()
    db.refresh(client)
    return {"id": client.id, "business_name": client.business_name}


@app.post("/api/campaigns/generate", tags=["Campaigns"])
async def generate_campaign(
    request: StrategyRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Basic flow: URL → strategy → campaign + role-based tasks → talent matching.
    For the full experience with granular weekly tasks, use /api/campaigns/generate-full instead.
    """
    # Step 1-2: AI analysis
    try:
        strategy = await ai_analyze_business(request.url, request.goals)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

    # Find or create client
    client = db.query(Client).filter(Client.website_url == request.url).first()
    if not client:
        domain = request.url.replace("https://", "").replace("http://", "").split("/")[0]
        client = Client(
            business_name=strategy.get("business_name", domain),
            website_url=request.url,
            contact_name="",
            contact_email="",
            business_analysis=strategy,
            industry=strategy.get("industry"),
            target_audience=strategy.get("target_audience"),
        )
        db.add(client)
        db.commit()
        db.refresh(client)

    # Create campaign
    campaign = Campaign(
        client_id=client.id,
        name=f"{strategy['business_name']} Growth Campaign",
        strategy=strategy,
        status="draft",
    )
    db.add(campaign)
    db.commit()
    db.refresh(campaign)

    # Create tasks from strategy team roles
    tasks_created = []
    for role in strategy.get("team_roles", []):
        calendar_items = [
            item for item in strategy.get("content_calendar", [])
            if item.get("channel", "").lower() in role.get("channel", "").lower()
            or role.get("channel", "").lower() in item.get("channel", "").lower()
        ]

        task_description = role["description"]
        if calendar_items:
            task_description += "\n\nContent schedule:\n" + "\n".join(
                f"- {item['day']}: {item['title']} ({item['format']})" for item in calendar_items
            )

        task = Task(
            campaign_id=campaign.id,
            title=f"{role['title']} — {role['channel']}",
            description=task_description,
            channel=role.get("channel"),
            required_role=role["title"],
            required_skills=role.get("skills", []),
            status="pending",
            due_date=datetime.utcnow() + timedelta(days=7),
            recurrence="weekly" if calendar_items else "one-time",
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        tasks_created.append({"id": task.id, "title": task.title, "role": role["title"]})

    # Queue background matching
    background_tasks.add_task(auto_match_and_assign_all_tasks, campaign.id)

    return {
        "campaign_id": campaign.id,
        "client_id": client.id,
        "strategy": strategy,
        "tasks_created": tasks_created,
        "status": "Campaign created. AI is now matching talent and will send briefs automatically.",
    }


# ──────────────────────────────────────────────
# Task Breakdown & Full Campaign Generation
# ──────────────────────────────────────────────

@app.post("/api/campaigns/{campaign_id}/breakdown", tags=["Campaigns"])
async def break_down_campaign(
    campaign_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Takes an existing campaign's strategy and breaks it into
    specific weekly tasks (12-24 tasks across 4 weeks).
    Replaces any existing pending tasks, then kicks off talent matching.
    """
    campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    strategy = campaign.strategy
    if not strategy:
        raise HTTPException(status_code=400, detail="Campaign has no strategy. Generate one first.")

    budget_context = strategy.get("estimated_monthly_budget")

    try:
        task_list = await ai_break_down_strategy(strategy, budget_context)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Task breakdown failed: {str(e)}")

    # Clear existing pending tasks (don't touch assigned/in-progress ones)
    db.query(Task).filter(
        Task.campaign_id == campaign_id,
        Task.status == "pending",
    ).delete()
    db.commit()

    # Create new tasks from AI breakdown
    created_tasks = []
    for t in task_list:
        week_num = t.get("week", 1)
        due_date = datetime.utcnow() + timedelta(weeks=week_num)

        task = Task(
            campaign_id=campaign_id,
            title=t["title"],
            description=t["description"],
            channel=t.get("channel"),
            required_role=t["required_role"],
            required_skills=t.get("required_skills", []),
            status="pending",
            priority=t.get("priority", 2),
            due_date=due_date,
            recurrence=t.get("recurrence", "one-time"),
        )
        db.add(task)
        db.commit()
        db.refresh(task)

        created_tasks.append({
            "id": task.id,
            "title": task.title,
            "channel": task.channel,
            "required_role": task.required_role,
            "week": week_num,
            "priority": t.get("priority", 2),
            "estimated_hours": t.get("estimated_hours"),
            "deliverable_format": t.get("deliverable_format"),
            "recurrence": t.get("recurrence", "one-time"),
            "due_date": due_date.isoformat(),
        })

    background_tasks.add_task(auto_match_and_assign_all_tasks, campaign_id)

    return {
        "campaign_id": campaign_id,
        "total_tasks": len(created_tasks),
        "weeks_covered": 4,
        "tasks_by_week": {
            f"week_{w}": [t for t in created_tasks if t["week"] == w]
            for w in range(1, 5)
        },
        "status": "Tasks created. AI is now matching talent to each task.",
    }


@app.post("/api/campaigns/generate-full", tags=["Campaigns"])
async def generate_full_campaign(
    request: StrategyRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    The complete flow in one call:
    1. AI analyzes business URL → generates strategy
    2. AI breaks strategy into weekly tasks (4 weeks, 12-24 tasks)
    3. Kicks off talent matching for all tasks

    This is what the frontend should call for the full experience.
    """
    # Step 1: Generate strategy
    try:
        strategy = await ai_analyze_business(request.url, request.goals)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Strategy generation failed: {str(e)}")

    # Find or create client
    client = db.query(Client).filter(Client.website_url == request.url).first()
    if not client:
        domain = request.url.replace("https://", "").replace("http://", "").split("/")[0]
        client = Client(
            business_name=strategy.get("business_name", domain),
            website_url=request.url,
            contact_name="",
            contact_email="",
            business_analysis=strategy,
            industry=strategy.get("industry"),
            target_audience=strategy.get("target_audience"),
        )
        db.add(client)
        db.commit()
        db.refresh(client)

    # Create campaign
    campaign = Campaign(
        client_id=client.id,
        name=f"{strategy['business_name']} Growth Campaign",
        strategy=strategy,
        status="draft",
    )
    db.add(campaign)
    db.commit()
    db.refresh(campaign)

    # Step 2: Break into weekly tasks
    budget_context = strategy.get("estimated_monthly_budget")
    try:
        task_list = await ai_break_down_strategy(strategy, budget_context)
    except Exception as e:
        return {
            "campaign_id": campaign.id,
            "client_id": client.id,
            "strategy": strategy,
            "tasks_by_week": {},
            "total_tasks": 0,
            "status": "Strategy generated but task breakdown failed. Call /api/campaigns/{id}/breakdown to retry.",
        }

    # Save tasks
    created_tasks = []
    for t in task_list:
        week_num = t.get("week", 1)
        due_date = datetime.utcnow() + timedelta(weeks=week_num)

        task = Task(
            campaign_id=campaign.id,
            title=t["title"],
            description=t["description"],
            channel=t.get("channel"),
            required_role=t["required_role"],
            required_skills=t.get("required_skills", []),
            status="pending",
            priority=t.get("priority", 2),
            due_date=due_date,
            recurrence=t.get("recurrence", "one-time"),
        )
        db.add(task)
        db.commit()
        db.refresh(task)

        created_tasks.append({
            "id": task.id,
            "title": task.title,
            "channel": task.channel,
            "required_role": task.required_role,
            "week": week_num,
            "priority": t.get("priority", 2),
            "estimated_hours": t.get("estimated_hours"),
            "deliverable_format": t.get("deliverable_format"),
            "due_date": due_date.isoformat(),
        })

    # Step 3: Match talent in background
    background_tasks.add_task(auto_match_and_assign_all_tasks, campaign.id)

    return {
        "campaign_id": campaign.id,
        "client_id": client.id,
        "strategy": strategy,
        "tasks_by_week": {
            f"week_{w}": [t for t in created_tasks if t["week"] == w]
            for w in range(1, 5)
        },
        "total_tasks": len(created_tasks),
        "status": "Full campaign generated. Strategy created, tasks broken down, talent matching in progress.",
    }


# ──────────────────────────────────────────────
# Background: Auto-match talent to tasks
# ──────────────────────────────────────────────

async def auto_match_and_assign_all_tasks(campaign_id: int):
    """Background job: match talent to all pending tasks in a campaign."""
    db = SessionLocal()
    try:
        tasks = db.query(Task).filter(
            Task.campaign_id == campaign_id,
            Task.status == "pending",
        ).all()

        for task in tasks:
            available = db.query(Talent).filter(
                Talent.status == "active",
                Talent.primary_role.ilike(f"%{task.required_role.split()[0]}%"),
            ).all()

            if not available:
                available = db.query(Talent).filter(Talent.status == "active").all()

            if not available:
                continue

            task_data = {
                "title": task.title,
                "description": task.description,
                "channel": task.channel,
                "required_role": task.required_role,
                "required_skills": task.required_skills,
            }
            talent_data = [
                {
                    "talent_id": t.id,
                    "name": t.name,
                    "primary_role": t.primary_role,
                    "channels": t.channels,
                    "skills": t.skills,
                    "hourly_rate": t.hourly_rate,
                    "per_deliverable_rate": t.per_deliverable_rate,
                    "avg_rating": t.avg_rating,
                    "total_tasks_completed": t.total_tasks_completed,
                    "availability_hours_per_week": t.availability_hours_per_week,
                }
                for t in available[:20]
            ]

            try:
                matches = await ai_match_talent(task_data, talent_data)
                if matches:
                    best = matches[0]
                    assignment = TaskAssignment(
                        task_id=task.id,
                        talent_id=best["talent_id"],
                        payment_amount=None,
                    )
                    db.add(assignment)
                    task.status = "assigned"
                    db.commit()
            except Exception as e:
                print(f"Matching error for task {task.id}: {e}")
                continue

    finally:
        db.close()


# ──────────────────────────────────────────────
# API Routes — Task Lifecycle
# ──────────────────────────────────────────────

@app.get("/api/campaigns/{campaign_id}/tasks", tags=["Tasks"])
def list_campaign_tasks(campaign_id: int, db: Session = Depends(get_db)):
    """List all tasks for a campaign with assignments."""
    tasks = db.query(Task).filter(Task.campaign_id == campaign_id).all()
    result = []
    for t in tasks:
        task_data = {
            "id": t.id,
            "title": t.title,
            "channel": t.channel,
            "required_role": t.required_role,
            "status": t.status,
            "due_date": t.due_date.isoformat() if t.due_date else None,
            "assigned_talent": None,
        }
        if t.assignment:
            talent = db.query(Talent).filter(Talent.id == t.assignment.talent_id).first()
            task_data["assigned_talent"] = {
                "id": talent.id,
                "name": talent.name,
                "role": talent.primary_role,
                "accepted": t.assignment.accepted,
            }
        result.append(task_data)
    return result


@app.post("/api/tasks/{task_id}/submit", tags=["Tasks"])
async def submit_deliverable(
    task_id: int,
    submission: TaskSubmission,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Talent submits a deliverable. AI reviews it before client sees it."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    task.deliverable_url = submission.deliverable_url
    task.status = "review"
    db.commit()

    background_tasks.add_task(ai_review_and_route, task_id, submission.deliverable_url)

    return {"status": "Submitted. AI is reviewing your deliverable."}


async def ai_review_and_route(task_id: int, deliverable_url: str):
    """Background: AI reviews deliverable and routes to client or back to talent."""
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            return

        campaign = db.query(Campaign).filter(Campaign.id == task.campaign_id).first()
        client = db.query(Client).filter(Client.id == campaign.client_id).first()

        task_data = {
            "title": task.title,
            "description": task.description,
            "channel": task.channel,
            "required_role": task.required_role,
        }
        client_context = {
            "business_name": client.business_name,
            "industry": client.industry,
            "target_audience": client.target_audience,
        }

        review = await ai_review_deliverable(task_data, deliverable_url, client_context)
        task.ai_review_notes = json.dumps(review)

        if review.get("recommendation") == "approve":
            task.status = "client_approval"
            db.commit()
        else:
            task.status = "revision_requested"
            db.commit()

    finally:
        db.close()


@app.post("/api/tasks/{task_id}/approve", tags=["Tasks"])
def client_approve_task(
    task_id: int,
    decision: ApprovalDecision,
    db: Session = Depends(get_db),
):
    """Client approves or requests revision on a deliverable."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if decision.approved:
        task.status = "completed"
        task.client_feedback = decision.feedback

        if task.assignment:
            talent = db.query(Talent).filter(Talent.id == task.assignment.talent_id).first()
            if talent:
                talent.total_tasks_completed += 1
                task.assignment.payment_status = "paid"
    else:
        task.status = "revision_requested"
        task.client_feedback = decision.feedback

    db.commit()
    return {"status": "approved" if decision.approved else "revision_requested"}


# ──────────────────────────────────────────────
# API Routes — Slack Integration
# ──────────────────────────────────────────────

@app.post("/api/slack/interactions", tags=["Slack"])
async def handle_slack_interaction(payload: dict):
    """Handles Slack interactive messages (button clicks for approve/reject)."""
    action = payload.get("actions", [{}])[0]
    action_id = action.get("action_id", "")
    task_id = int(action.get("value", 0))

    if "approve" in action_id:
        db = SessionLocal()
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = "completed"
            if task.assignment:
                talent = db.query(Talent).filter(Talent.id == task.assignment.talent_id).first()
                if talent:
                    talent.total_tasks_completed += 1
            db.commit()
        db.close()
        return {"response_action": "update", "text": "✅ Approved! Your team is scheduling publication."}

    elif "revision" in action_id:
        db = SessionLocal()
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = "revision_requested"
            db.commit()
        db.close()
        return {"response_action": "update", "text": "🔄 Revision requested. Your team has been notified."}

    return {"status": "ok"}


# ──────────────────────────────────────────────
# API Routes — Dashboard / Overview
# ──────────────────────────────────────────────

@app.get("/api/campaigns/{campaign_id}/dashboard", tags=["Dashboard"])
def campaign_dashboard(campaign_id: int, db: Session = Depends(get_db)):
    """Overview dashboard for a campaign — what the client sees."""
    campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    tasks = db.query(Task).filter(Task.campaign_id == campaign_id).all()

    status_counts = {}
    for t in tasks:
        status_counts[t.status] = status_counts.get(t.status, 0) + 1

    pending_approvals = [
        {
            "task_id": t.id,
            "title": t.title,
            "channel": t.channel,
            "deliverable_url": t.deliverable_url,
            "ai_review": json.loads(t.ai_review_notes) if t.ai_review_notes else None,
        }
        for t in tasks
        if t.status == "client_approval"
    ]

    return {
        "campaign_name": campaign.name,
        "status": campaign.status,
        "task_summary": status_counts,
        "total_tasks": len(tasks),
        "pending_approvals": pending_approvals,
        "team": [
            {
                "task": t.title,
                "talent": (
                    db.query(Talent).filter(Talent.id == t.assignment.talent_id).first().name
                    if t.assignment else "Matching..."
                ),
                "status": t.status,
            }
            for t in tasks
        ],
    }


# ──────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "fleetwork-backend", "version": "0.2.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
