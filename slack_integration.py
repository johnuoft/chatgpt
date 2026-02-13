"""
GrowthTeam Slack Integration
==============================
Handles all Slack communication:
- Client approval messages (approve/reject buttons)
- Status updates to client channel
- Weekly summary reports

Setup:
1. Create a Slack App at api.slack.com/apps
2. Enable "Interactivity" and set Request URL to your /api/slack/interactions endpoint
3. Add Bot Token Scopes: chat:write, im:write, channels:read
4. Install to workspace and grab the Bot Token
5. Set SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET env vars
"""

import os
import json
import hashlib
import hmac
import time
from typing import Optional

import httpx

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")
SLACK_API_BASE = "https://slack.com/api"


async def send_slack_message(channel_id: str, text: str, blocks: list = None) -> dict:
    """Send a message to a Slack channel."""
    async with httpx.AsyncClient() as client:
        payload = {
            "channel": channel_id,
            "text": text,
        }
        if blocks:
            payload["blocks"] = blocks

        response = await client.post(
            f"{SLACK_API_BASE}/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            json=payload,
        )
        return response.json()


async def send_approval_message(
    channel_id: str,
    task_id: int,
    task_title: str,
    talent_name: str,
    channel_name: str,
    deliverable_url: str,
    ai_summary: str,
    ai_quality_score: float,
):
    """
    Send an approval request to the client's Slack channel.
    This is the ONLY thing the client needs to interact with.
    """
    quality_emoji = "🟢" if ai_quality_score >= 7 else "🟡" if ai_quality_score >= 5 else "🔴"

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"📋 New deliverable ready for review",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*{task_title}*\n"
                    f"👤 {talent_name} · 📱 {channel_name}\n"
                    f"{quality_emoji} AI Quality Score: {ai_quality_score}/10\n\n"
                    f"_{ai_summary}_"
                ),
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"<{deliverable_url}|📎 View Deliverable>",
            },
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅ Approve"},
                    "style": "primary",
                    "action_id": f"approve_task",
                    "value": str(task_id),
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "🔄 Request Revision"},
                    "action_id": f"revision_task",
                    "value": str(task_id),
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "💬 Comment"},
                    "action_id": f"comment_task",
                    "value": str(task_id),
                },
            ],
        },
        {"type": "divider"},
    ]

    return await send_slack_message(
        channel_id=channel_id,
        text=f"New deliverable from {talent_name}: {task_title}",
        blocks=blocks,
    )


async def send_status_update(channel_id: str, campaign_name: str, updates: list[dict]):
    """Send a daily/weekly status update to the client."""
    update_lines = []
    for u in updates:
        status_emoji = {
            "completed": "✅",
            "in_progress": "🔨",
            "client_approval": "👀",
            "assigned": "📋",
            "pending": "⏳",
        }.get(u["status"], "⏳")
        update_lines.append(f"{status_emoji} *{u['title']}* — {u['talent']} — _{u['status']}_")

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"📊 {campaign_name} — Weekly Update"},
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "\n".join(update_lines),
            },
        },
    ]

    return await send_slack_message(channel_id, f"Weekly update: {campaign_name}", blocks)


async def send_campaign_launch_notification(
    channel_id: str,
    campaign_name: str,
    team_roles: list[dict],
    timeline: str,
    budget: str,
):
    """Notify client that their campaign has been set up and talent is being matched."""
    team_lines = [f"• *{r['title']}* ({r['channel']}) — matching talent..." for r in team_roles]

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "🚀 Your GrowthTeam campaign is live!"},
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*{campaign_name}*\n"
                    f"Budget: {budget} · Timeline: {timeline}\n\n"
                    f"Your AI project manager is assembling your team:\n"
                    + "\n".join(team_lines)
                    + "\n\nI'll ping you here when deliverables are ready for approval. "
                    "That's all you need to do — just approve or request changes. ✌️"
                ),
            },
        },
    ]

    return await send_slack_message(channel_id, f"Campaign launched: {campaign_name}", blocks)


def verify_slack_signature(timestamp: str, body: str, signature: str) -> bool:
    """Verify incoming Slack requests are genuine."""
    if abs(time.time() - float(timestamp)) > 60 * 5:
        return False  # request too old

    sig_basestring = f"v0:{timestamp}:{body}"
    my_sig = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(),
        sig_basestring.encode(),
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(my_sig, signature)
