"""
intervention_service.py
------------------------
Persists which AUF Early Intervention Policy checklist items have been
completed for each student (see utils/intervention_policy.py for the
policy content itself). Backed by the `intervention_log` table, keyed
by Student_ID rather than a foreign key to students.id, since the
students table is replaced wholesale on every dataset upload - this
keeps a teacher's tracked progress intact across re-uploads as long as
the same Student_ID reappears.
"""

from datetime import datetime

from sqlalchemy import text

from utils.db import get_engine


def get_actions_for_student(student_id):
    """Returns {action_key: {completed, completed_at, completed_by_name, notes}}."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT il.action_key, il.completed, il.completed_at, il.notes,
                       u.full_name AS completed_by_name
                FROM intervention_log il
                LEFT JOIN users u ON il.completed_by = u.id
                WHERE il.student_id = :sid
                """
            ),
            {"sid": student_id},
        ).mappings().all()
    out = {}
    for r in rows:
        item = dict(r)
        if isinstance(item.get("completed_at"), str):        # some drivers return text
            try:
                item["completed_at"] = datetime.fromisoformat(item["completed_at"])
            except ValueError:
                item["completed_at"] = None
        out[item["action_key"]] = item
    return out


def toggle_action(student_id, stage, action_key, user_id):
    """
    Flips the completed state for one checklist item, creating the row
    the first time it's touched. Returns the new completed state.
    """
    engine = get_engine()
    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT id, completed FROM intervention_log WHERE student_id=:sid AND action_key=:ak"),
            {"sid": student_id, "ak": action_key},
        ).mappings().first()

        if existing:
            new_state = not existing["completed"]
            conn.execute(
                text(
                    """
                    UPDATE intervention_log
                    SET completed=:completed, completed_at=:completed_at, completed_by=:uid
                    WHERE id=:id
                    """
                ),
                {
                    "completed": new_state,
                    "completed_at": datetime.now() if new_state else None,
                    "uid": user_id if new_state else None,
                    "id": existing["id"],
                },
            )
            return new_state
        else:
            conn.execute(
                text(
                    """
                    INSERT INTO intervention_log (student_id, stage, action_key, completed, completed_at, completed_by)
                    VALUES (:sid, :stage, :ak, TRUE, :now, :uid)
                    """
                ),
                {"sid": student_id, "stage": stage, "ak": action_key, "now": datetime.now(), "uid": user_id},
            )
            return True


def get_progress_summary(student_id, stage_actions):
    """
    stage_actions: {stage_number: [(action_key, label), ...]}
    Returns {stage_number: (completed_count, total_count)}.
    """
    done = get_actions_for_student(student_id)
    summary = {}
    for stage_num, actions in stage_actions.items():
        total = len(actions)
        completed = sum(1 for key, _ in actions if done.get(key, {}).get("completed"))
        summary[stage_num] = (completed, total)
    return summary
