# ⚡ GrowthTeam Backend

**AI-managed marketing agency platform.** Real talent. AI orchestration. Client just approves.

---

## Architecture

```
Client (Slack)           Talent (Email/Dashboard)
    │                           │
    │  approve / reject         │  accept / submit deliverables
    │                           │
    ▼                           ▼
┌─────────────────────────────────────────┐
│           GrowthTeam Backend            │
│                                         │
│  ┌──────────┐  ┌───────────────────┐    │
│  │ Strategy  │  │  AI Project Mgr   │    │
│  │ Generator │  │  (Claude API)     │    │
│  │ (Claude)  │  │                   │    │
│  └────┬─────┘  │ • Generate briefs  │    │
│       │        │ • Review work      │    │
│       ▼        │ • Route approvals  │    │
│  ┌──────────┐  │ • Send feedback    │    │
│  │ Matching  │  │ • Schedule content│    │
│  │ Engine    │  └───────────────────┘    │
│  │ (Claude)  │                           │
│  └────┬─────┘  ┌───────────────────┐    │
│       │        │   Talent Database  │    │
│       ▼        │   200+ specialists │    │
│  ┌──────────┐  └───────────────────┘    │
│  │   Task   │                           │
│  │ Manager  │  ┌───────────────────┐    │
│  │          │  │  Slack Integration │    │
│  └──────────┘  │  (Client approvals)│    │
│                └───────────────────┘    │
└─────────────────────────────────────────┘
```

## The Flow

```
1. Client pastes URL → AI analyzes business
2. AI generates strategy + recommends team roles
3. Client approves plan
4. AI auto-matches talent from database
5. AI sends briefs to talent via email
6. Talent creates content and submits
7. AI reviews deliverables (quality check)
8. If good → sends to client via Slack for approval
9. If needs work → sends revision request to talent
10. Client approves from Slack (one button click)
11. Content gets scheduled / published
12. Repeat weekly
```

## Files

```
growthteam-backend/
├── app.py                 # Main FastAPI application (all routes + DB models)
├── slack_integration.py   # Slack bot for client approvals
├── email_templates.py     # Branded email templates for talent communication
├── import_talent.py       # Script to bulk import your 200+ talent
├── sample_talent.csv      # Template CSV — fill this in with your talent
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
└── README.md              # This file
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# 3. Run the server
python app.py
# Server starts at http://localhost:8000
# API docs at http://localhost:8000/docs

# 4. Import your talent
python import_talent.py --file your_talent.csv

# 5. Test the flow
curl -X POST http://localhost:8000/api/campaigns/generate \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

## API Endpoints

### Talent
- `POST /api/talent` — Add single talent
- `POST /api/talent/bulk` — Bulk import (for your 200+ list)
- `GET /api/talent` — List talent (filter by role, channel)
- `GET /api/talent/{id}` — Get talent profile

### Campaigns
- `POST /api/campaigns/generate` — **The core flow**: URL → strategy → tasks → auto-match
- `GET /api/campaigns/{id}/tasks` — List all tasks with assignments
- `GET /api/campaigns/{id}/dashboard` — Client dashboard view

### Task Lifecycle
- `POST /api/tasks/{id}/submit` — Talent submits deliverable
- `POST /api/tasks/{id}/approve` — Client approves/rejects

### Slack
- `POST /api/slack/interactions` — Handles Slack button clicks

---

## What YOU Need To Do (Action Plan)

### Week 1: Foundation
- [ ] **Get your Anthropic API key** → console.anthropic.com (you need this for all AI features)
- [ ] **Set up a Slack workspace** for GrowthTeam (or use your existing one)
- [ ] **Create a Slack App** at api.slack.com/apps (follow .env.example instructions)
- [ ] **Sign up for Resend** (resend.com) for transactional emails — it's free up to 3K emails/month
- [ ] **Fill in sample_talent.csv** with your 200+ UGC creators and editors
- [ ] **Run the import**: `python import_talent.py --file your_talent.csv`

### Week 2: Deploy & Test
- [ ] **Deploy to Railway or Render** (easiest for Python/FastAPI)
  - Railway: `railway init` → `railway up` (connects to GitHub)
  - Render: Connect repo → auto-deploys
- [ ] **Switch from SQLite to PostgreSQL** (Railway/Render give you a free Postgres DB)
  - Just change DATABASE_URL in your .env
- [ ] **Test the full flow** with 2-3 real clients:
  1. They paste their URL
  2. AI generates strategy
  3. You manually verify the talent matches make sense
  4. Talent gets the brief email
  5. You check Slack approvals work

### Week 3: Polish & Launch
- [ ] **Connect your frontend** (the React prototype) to this backend
- [ ] **Set up Stripe** for payments (talent payouts + client billing)
- [ ] **Add a talent onboarding form** on your site so new talent can apply
- [ ] **Set up error monitoring** with Sentry
- [ ] **Soft launch** with 5-10 clients from your network

### Later: Scale
- [ ] Add social media API integrations (Meta, TikTok, YouTube) for auto-publishing
- [ ] Build a proper talent dashboard where they can see assignments + submit work
- [ ] Add analytics/reporting for clients
- [ ] Switch BackgroundTasks to Celery + Redis for production job queue
- [ ] Add Stripe Connect for automated talent payouts

---

## Key Decisions You'll Need to Make

1. **Pricing model**: Per-campaign? Monthly subscription? Percentage markup on talent rates?
2. **Talent payment**: When do talent get paid? On approval? Net-30? How much do you take?
3. **Quality control**: How much does the AI auto-approve vs. what goes to clients?
4. **Scope**: Start with just UGC/video (your strength) or go multi-channel from day 1?

**My recommendation**: Start with UGC/video only on TikTok + Instagram. Charge clients a flat monthly fee ($2K-5K/mo) that covers talent costs + your margin. Pay talent per deliverable within 7 days of client approval. Let the AI auto-approve anything scoring 7+/10 and only surface lower-quality work to you for manual review before it goes to the client.
