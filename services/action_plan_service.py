"""
action_plan_service.py
------------------------
The "Teacher Action Plan" feature: reminders/scheduled follow-ups a
teacher creates for themselves (optionally tied to a specific student),
each with a status of Pending, Scheduled, Ongoing, or Finished.

Every function here is scoped to one teacher (created_by) - a teacher
only ever sees or modifies their own action plan items, consistent with
the same "only see/edit what's yours" rule applied to student data.
"""

from datetime import date, datetime, timedelta

from sqlalchemy import text

from utils.db import get_engine

STATUSES = ["Pending", "Scheduled", "Ongoing", "Finished"]
PRIORITIES = ["High", "Medium", "Low"]

STATUS_BADGE_CLASS = {
    "Pending": "badge-warning",
    "Scheduled": "badge-info",
    "Ongoing": "badge-info",
    "Finished": "badge-success",
}


def list_items(teacher_id):
    """This teacher's action plan items, soonest due date first."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, student_id, title, notes, priority, due_date, status, created_at
                FROM action_plans
                WHERE created_by = :tid
                ORDER BY
                    CASE status WHEN 'Finished' THEN 1 ELSE 0 END ASC,
                    (due_date IS NULL) ASC,
                    due_date ASC,
                    created_at DESC
                """
            ),
            {"tid": teacher_id},
        ).mappings().all()

    items = []
    today = date.today()
    for r in rows:
        item = dict(r)
        due = item.get("due_date")
        if isinstance(due, str):                               # some drivers return text
            try:
                due = date.fromisoformat(due[:10])
            except ValueError:
                due = None
            item["due_date"] = due
        item["badge_class"] = STATUS_BADGE_CLASS.get(item["status"], "badge-warning")
        item["is_overdue"] = bool(due and due < today and item["status"] != "Finished")
        item["is_due_soon"] = bool(
            due and item["status"] != "Finished" and today <= due <= today + timedelta(days=7)
        )
        items.append(item)
    return items


def counts_by_status(teacher_id):
    """{'Pending': n, 'Scheduled': n, 'Ongoing': n, 'Finished': n}"""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT status, COUNT(*) AS n FROM action_plans WHERE created_by = :tid GROUP BY status"),
            {"tid": teacher_id},
        ).mappings().all()
    counts = {s: 0 for s in STATUSES}
    for r in rows:
        if r["status"] in counts:
            counts[r["status"]] = r["n"]
    return counts


def upcoming_alert_count(teacher_id):
    """
    Number of not-yet-finished items that are overdue or due within 7
    days - powers the small dashboard notification banner.
    """
    engine = get_engine()
    with engine.connect() as conn:
        count = conn.execute(
            text(
                """
                SELECT COUNT(*) FROM action_plans
                WHERE created_by = :tid
                AND status != 'Finished'
                AND due_date IS NOT NULL
                AND due_date <= :horizon
                """
            ),
            {"tid": teacher_id, "horizon": date.today() + timedelta(days=7)},
        ).scalar()
    return int(count or 0)


def create_item(teacher_id, title, student_id=None, notes=None, priority="Medium", due_date=None, status="Pending"):
    if not title or not title.strip():
        return False
    if priority not in PRIORITIES:
        priority = "Medium"
    if status not in STATUSES:
        status = "Pending"

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO action_plans (created_by, student_id, title, notes, priority, due_date, status)
                VALUES (:tid, :sid, :title, :notes, :priority, :due_date, :status)
                """
            ),
            {
                "tid": teacher_id,
                "sid": student_id or None,
                "title": title.strip(),
                "notes": (notes or "").strip() or None,
                "priority": priority,
                "due_date": due_date or None,
                "status": status,
            },
        )
    return True


def update_status(teacher_id, item_id, status):
    """Ownership-enforced: the WHERE clause itself checks created_by."""
    if status not in STATUSES:
        return False
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            text(
                """
                UPDATE action_plans SET status = :status, updated_at = :now
                WHERE id = :id AND created_by = :tid
                """
            ),
            {"status": status, "now": datetime.now(), "id": item_id, "tid": teacher_id},
        )
        return result.rowcount > 0


def delete_item(teacher_id, item_id):
    """Ownership-enforced: the WHERE clause itself checks created_by."""
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            text("DELETE FROM action_plans WHERE id = :id AND created_by = :tid"),
            {"id": item_id, "tid": teacher_id},
        )
        return result.rowcount > 0
