"""LOCAL administrator CLI for reviewing submissions, events and email outbox.

Examples:
  python admin.py list
  python admin.py list --status pending
  python admin.py show abcdef123456
  python admin.py review abcdef123456 needs_info --note "Podaj numer modelu"
  python admin.py outbox
  python admin.py logs
"""
import argparse
import json
from src.review_store import (
    STATUSES, change_status, get_admin_case, list_cases, list_events, list_outbox
)


def main():
    parser = argparse.ArgumentParser(description="Local moderation console (not a web admin API)")
    sub = parser.add_subparsers(dest="command", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--status", choices=STATUSES)
    sub.add_parser("logs")
    sub.add_parser("outbox")
    show = sub.add_parser("show")
    show.add_argument("id")
    review = sub.add_parser("review")
    review.add_argument("id")
    review.add_argument("status", choices=STATUSES)
    review.add_argument("--note", default="")
    args = parser.parse_args()

    if args.command == "list":
        data = list_cases(args.status)
    elif args.command == "logs":
        data = list_events()
    elif args.command == "outbox":
        data = list_outbox()
    elif args.command == "show":
        data = get_admin_case(args.id)
    else:
        change_status(case_id=args.id, new_status=args.status, reviewer_note=args.note)
        data = {"updated": args.id, "status": args.status}
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
