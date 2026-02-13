"""
GrowthTeam Talent Importer
============================
Use this script to import your 200+ talent from a CSV or JSON file.

USAGE:
  python import_talent.py --file talent.csv
  python import_talent.py --file talent.json

CSV FORMAT (talent.csv):
  name,email,primary_role,channels,skills,hourly_rate,per_deliverable_rate,rate_notes,portfolio_url,availability_hours_per_week,timezone
  "Jane Smith","jane@email.com","UGC Creator","TikTok,Instagram","short-form video,storytelling",0,60,"$50-80/video","https://jane.portfolio.com",15,"EST"
  "Alex Johnson","alex@email.com","Video Editor","TikTok,YouTube","video editing,motion graphics",45,0,"$45/hr","https://alex.work",20,"PST"

JSON FORMAT (talent.json):
  [
    {
      "name": "Jane Smith",
      "email": "jane@email.com",
      "primary_role": "UGC Creator",
      "channels": ["TikTok", "Instagram"],
      "skills": ["short-form video", "storytelling"],
      "per_deliverable_rate": 60,
      "rate_notes": "$50-80/video",
      "portfolio_url": "https://jane.portfolio.com",
      "availability_hours_per_week": 15,
      "timezone": "EST"
    }
  ]

NOTES:
- Duplicate emails are automatically skipped
- channels and skills are comma-separated in CSV, arrays in JSON
- Set hourly_rate OR per_deliverable_rate (or both)
"""

import argparse
import csv
import json
import sys
import httpx

API_BASE = "http://localhost:8000"


def parse_csv(filepath: str) -> list[dict]:
    """Parse talent from CSV file."""
    talents = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            talent = {
                "name": row.get("name", "").strip(),
                "email": row.get("email", "").strip(),
                "primary_role": row.get("primary_role", "").strip(),
                "channels": [c.strip() for c in row.get("channels", "").split(",") if c.strip()],
                "skills": [s.strip() for s in row.get("skills", "").split(",") if s.strip()],
                "hourly_rate": float(row.get("hourly_rate", 0)) or None,
                "per_deliverable_rate": float(row.get("per_deliverable_rate", 0)) or None,
                "rate_notes": row.get("rate_notes", "").strip() or None,
                "portfolio_url": row.get("portfolio_url", "").strip() or None,
                "availability_hours_per_week": int(row.get("availability_hours_per_week", 20)),
                "timezone": row.get("timezone", "EST").strip(),
                "preferred_contact": row.get("preferred_contact", "email").strip(),
            }
            if talent["name"] and talent["email"]:
                talents.append(talent)
    return talents


def parse_json(filepath: str) -> list[dict]:
    """Parse talent from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Ensure all required fields
    talents = []
    for item in data:
        talent = {
            "name": item.get("name", ""),
            "email": item.get("email", ""),
            "primary_role": item.get("primary_role", ""),
            "channels": item.get("channels", []),
            "skills": item.get("skills", []),
            "hourly_rate": item.get("hourly_rate"),
            "per_deliverable_rate": item.get("per_deliverable_rate"),
            "rate_notes": item.get("rate_notes"),
            "portfolio_url": item.get("portfolio_url"),
            "bio": item.get("bio"),
            "availability_hours_per_week": item.get("availability_hours_per_week", 20),
            "timezone": item.get("timezone", "EST"),
            "preferred_contact": item.get("preferred_contact", "email"),
        }
        if talent["name"] and talent["email"] and talent["primary_role"]:
            talents.append(talent)
    return talents


def import_to_api(talents: list[dict], api_base_url: str = API_BASE):
    """Send talents to the GrowthTeam API via bulk import."""
    print(f"\n📋 Importing {len(talents)} talent profiles...\n")

    response = httpx.post(
        f"{api_base_url}/api/talent/bulk",
        json={"talents": talents},
        timeout=30,
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Created: {result['created']}")
        print(f"⏭️  Skipped (duplicate email): {result['skipped']}")
        if result["skipped_emails"]:
            print(f"   Skipped emails: {', '.join(result['skipped_emails'][:10])}")
            if len(result["skipped_emails"]) > 10:
                print(f"   ... and {len(result['skipped_emails']) - 10} more")
    else:
        print(f"❌ Error: {response.status_code} — {response.text}")


def main():
    parser = argparse.ArgumentParser(description="Import talent to GrowthTeam")
    parser.add_argument("--file", required=True, help="Path to CSV or JSON file")
    parser.add_argument("--api", default=API_BASE, help="API base URL")
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate only, don't import")
    args = parser.parse_args()

    api_url = args.api

    filepath = args.file
    if filepath.endswith(".csv"):
        talents = parse_csv(filepath)
    elif filepath.endswith(".json"):
        talents = parse_json(filepath)
    else:
        print("❌ Unsupported file format. Use .csv or .json")
        sys.exit(1)

    print(f"📄 Parsed {len(talents)} talent profiles from {filepath}")

    # Show preview
    print("\n--- Preview (first 3) ---")
    for t in talents[:3]:
        print(f"  {t['name']} | {t['primary_role']} | {', '.join(t['channels'])} | {t['email']}")
    if len(talents) > 3:
        print(f"  ... and {len(talents) - 3} more\n")

    if args.dry_run:
        print("🔍 Dry run complete. No data imported.")
        return

    import_to_api(talents, api_url)


if __name__ == "__main__":
    main()
