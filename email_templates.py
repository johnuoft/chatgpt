"""
GrowthTeam Email Templates
============================
AI-generated emails sent to talent for:
- Task assignment / briefs
- Revision requests
- Payment confirmations
- Onboarding welcome

In production, use SendGrid, Resend, or AWS SES.
"""

import os
from typing import Optional
import httpx

# Use Resend, SendGrid, or any transactional email service
EMAIL_API_KEY = os.getenv("EMAIL_API_KEY", "")
FROM_EMAIL = "team@growthteam.ai"
FROM_NAME = "GrowthTeam AI"


async def send_email(to: str, subject: str, html_body: str, reply_to: str = None):
    """Send an email via your transactional email provider."""
    # Example using Resend (swap for SendGrid/SES as needed)
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {EMAIL_API_KEY}"},
            json={
                "from": f"{FROM_NAME} <{FROM_EMAIL}>",
                "to": [to],
                "subject": subject,
                "html": html_body,
                "reply_to": reply_to or FROM_EMAIL,
            },
        )
        return response.json()


def task_assignment_email(
    talent_name: str,
    task_title: str,
    brief: str,
    due_date: str,
    payment_amount: str,
    submit_url: str,
) -> tuple[str, str]:
    """Generate task assignment email for talent."""
    subject = f"New Assignment: {task_title}"
    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 600px; margin: 0 auto; background: #0a0a0a; color: #e5e5e5; border-radius: 12px; overflow: hidden;">
        <div style="background: #f97316; padding: 24px 32px;">
            <h1 style="margin: 0; font-size: 20px; color: #fff;">⚡ GrowthTeam</h1>
        </div>
        <div style="padding: 32px;">
            <p style="color: #999; margin-top: 0;">Hey {talent_name},</p>
            
            <h2 style="color: #f97316; font-size: 18px; margin-bottom: 8px;">New Assignment</h2>
            <h3 style="color: #e5e5e5; margin-top: 0;">{task_title}</h3>
            
            <div style="background: #111; border: 1px solid #1e1e1e; border-radius: 8px; padding: 20px; margin: 20px 0;">
                <p style="color: #ccc; margin: 0; white-space: pre-wrap; line-height: 1.7;">{brief}</p>
            </div>
            
            <div style="display: flex; gap: 24px; margin: 24px 0;">
                <div>
                    <span style="color: #666; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Due Date</span>
                    <p style="color: #e5e5e5; font-weight: 600; margin: 4px 0 0;">{due_date}</p>
                </div>
                <div>
                    <span style="color: #666; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Payment</span>
                    <p style="color: #f97316; font-weight: 600; margin: 4px 0 0;">{payment_amount}</p>
                </div>
            </div>
            
            <a href="{submit_url}" style="display: inline-block; background: #f97316; color: #fff; text-decoration: none; padding: 14px 28px; border-radius: 8px; font-weight: 600; margin-top: 16px;">Accept & View Brief →</a>
            
            <p style="color: #555; font-size: 13px; margin-top: 32px;">
                This assignment is managed by GrowthTeam's AI project manager. 
                Reply to this email if you have questions.
            </p>
        </div>
    </div>
    """
    return subject, html


def revision_request_email(
    talent_name: str,
    task_title: str,
    feedback: str,
    resubmit_url: str,
) -> tuple[str, str]:
    """Generate revision request email for talent."""
    subject = f"Revision Requested: {task_title}"
    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 600px; margin: 0 auto; background: #0a0a0a; color: #e5e5e5; border-radius: 12px; overflow: hidden;">
        <div style="background: #f97316; padding: 24px 32px;">
            <h1 style="margin: 0; font-size: 20px; color: #fff;">⚡ GrowthTeam</h1>
        </div>
        <div style="padding: 32px;">
            <p style="color: #999; margin-top: 0;">Hey {talent_name},</p>
            
            <h2 style="color: #eab308; font-size: 18px; margin-bottom: 8px;">🔄 Revision Requested</h2>
            <h3 style="color: #e5e5e5; margin-top: 0;">{task_title}</h3>
            
            <div style="background: #111; border: 1px solid #1e1e1e; border-radius: 8px; padding: 20px; margin: 20px 0;">
                <p style="color: #666; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-top: 0;">Feedback</p>
                <p style="color: #ccc; margin: 8px 0 0; line-height: 1.7;">{feedback}</p>
            </div>
            
            <a href="{resubmit_url}" style="display: inline-block; background: #f97316; color: #fff; text-decoration: none; padding: 14px 28px; border-radius: 8px; font-weight: 600; margin-top: 16px;">Resubmit Deliverable →</a>
            
            <p style="color: #555; font-size: 13px; margin-top: 32px;">
                Please resubmit within 48 hours. Reply if you need clarification.
            </p>
        </div>
    </div>
    """
    return subject, html


def payment_confirmation_email(
    talent_name: str,
    task_title: str,
    amount: str,
    payment_method: str = "bank transfer",
) -> tuple[str, str]:
    """Generate payment confirmation email."""
    subject = f"Payment Sent: {amount} for {task_title}"
    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 600px; margin: 0 auto; background: #0a0a0a; color: #e5e5e5; border-radius: 12px; overflow: hidden;">
        <div style="background: #f97316; padding: 24px 32px;">
            <h1 style="margin: 0; font-size: 20px; color: #fff;">⚡ GrowthTeam</h1>
        </div>
        <div style="padding: 32px;">
            <p style="color: #999; margin-top: 0;">Hey {talent_name},</p>
            
            <h2 style="color: #22c55e; font-size: 18px; margin-bottom: 8px;">💰 Payment Sent</h2>
            
            <div style="background: #111; border: 1px solid #1e1e1e; border-radius: 8px; padding: 20px; margin: 20px 0; text-align: center;">
                <p style="color: #22c55e; font-size: 32px; font-weight: 800; margin: 0;">{amount}</p>
                <p style="color: #666; margin: 8px 0 0;">{task_title}</p>
                <p style="color: #555; font-size: 13px; margin: 4px 0 0;">via {payment_method}</p>
            </div>
            
            <p style="color: #888; font-size: 14px;">
                Great work on this one! Payment has been processed and should 
                arrive within 1-3 business days.
            </p>
        </div>
    </div>
    """
    return subject, html


def talent_welcome_email(talent_name: str, dashboard_url: str) -> tuple[str, str]:
    """Welcome email when talent joins the platform."""
    subject = "Welcome to GrowthTeam ⚡"
    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 600px; margin: 0 auto; background: #0a0a0a; color: #e5e5e5; border-radius: 12px; overflow: hidden;">
        <div style="background: #f97316; padding: 24px 32px;">
            <h1 style="margin: 0; font-size: 20px; color: #fff;">⚡ GrowthTeam</h1>
        </div>
        <div style="padding: 32px;">
            <h2 style="color: #e5e5e5; margin-top: 0;">Welcome, {talent_name}!</h2>
            
            <p style="color: #999; line-height: 1.7;">
                You've been added to GrowthTeam's talent network. Here's how it works:
            </p>
            
            <div style="margin: 24px 0;">
                <div style="display: flex; align-items: flex-start; gap: 12px; margin-bottom: 16px;">
                    <span style="background: #f97316; color: #fff; width: 28px; height: 28px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700; flex-shrink: 0;">1</span>
                    <p style="color: #ccc; margin: 2px 0 0;"><strong>You get matched.</strong> Our AI matches you with clients based on your skills, channels, and availability.</p>
                </div>
                <div style="display: flex; align-items: flex-start; gap: 12px; margin-bottom: 16px;">
                    <span style="background: #f97316; color: #fff; width: 28px; height: 28px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700; flex-shrink: 0;">2</span>
                    <p style="color: #ccc; margin: 2px 0 0;"><strong>You get a brief.</strong> Each assignment comes with a clear brief — what to create, brand context, specs, and deadline.</p>
                </div>
                <div style="display: flex; align-items: flex-start; gap: 12px; margin-bottom: 16px;">
                    <span style="background: #f97316; color: #fff; width: 28px; height: 28px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700; flex-shrink: 0;">3</span>
                    <p style="color: #ccc; margin: 2px 0 0;"><strong>You deliver & get paid.</strong> Submit your work, it gets reviewed, and payment is processed automatically.</p>
                </div>
            </div>
            
            <p style="color: #999; line-height: 1.7;">
                No chasing invoices, no scope creep, no back-and-forth with clients. 
                Our AI handles all the project management so you can focus on doing great work.
            </p>
            
            <a href="{dashboard_url}" style="display: inline-block; background: #f97316; color: #fff; text-decoration: none; padding: 14px 28px; border-radius: 8px; font-weight: 600; margin-top: 16px;">View Your Dashboard →</a>
        </div>
    </div>
    """
    return subject, html
