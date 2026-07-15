"""Notion run recorder: every research run reconstructable from Notion (US-027).

The orchestrator's single Notion write funnel (docs/reference/notion-schema.md):
saved directives land in Research Ideas, each proposed hypothesis and the
operator's action on it in Hypothesis Log, each completed experiment's
backtest metrics in Backtest Results, and deliberate trading decisions
(promotion, halt/resume) in Decision Log. Database ids come from
orchestrator/config.yaml (written by ops/bootstrap_notion.py — never hardcode
them).

Recording is best-effort BY DESIGN: every public ``record_*`` method catches
and logs its own failures instead of raising, because a Notion outage must
never break the Slack conversation, the hypothesis poller, or a running
research loop. Page-id mappings (thread_ts -> idea page, interaction key ->
hypothesis row) persist in SQLite (StateStore ``notion_pages``), so later
lifecycle points update the same page across restarts.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from orchestrator import summary
from orchestrator.notion_client import NotionClient
from orchestrator.state import Directive, StateStore

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

# notion_pages.kind values (StateStore mapping table).
PAGE_KIND_IDEA = "idea"
PAGE_KIND_HYPOTHESIS = "hypothesis"

# Notion caps a single rich_text/title text object at 2000 characters.
_MAX_TEXT = 2000
_MAX_TITLE = 120

_T = TypeVar("_T")


class RecorderConfigError(RuntimeError):
    """orchestrator/config.yaml is missing the Notion database ids."""


@dataclass(frozen=True)
class NotionDatabases:
    """The six database ids bootstrap_notion.py writes into config.yaml."""

    research_ideas: str
    hypothesis_log: str
    backtest_results: str
    decision_log: str
    trade_ledger: str
    account_snapshots: str


def load_notion_databases(config_path: Path = DEFAULT_CONFIG_PATH) -> NotionDatabases:
    """Read the database ids from orchestrator/config.yaml."""
    import yaml

    try:
        loaded = yaml.safe_load(config_path.read_text())
    except OSError as exc:
        raise RecorderConfigError(
            f"cannot read {config_path} ({exc}) — run ops/bootstrap_notion.py to"
            " create the Notion databases and write their ids"
        ) from exc
    databases: Any = {}
    if isinstance(loaded, dict):
        databases = loaded.get("notion", {}).get("databases", {})
    if not isinstance(databases, dict):
        databases = {}
    missing = [f.name for f in fields(NotionDatabases) if not databases.get(f.name)]
    if missing:
        raise RecorderConfigError(
            f"{config_path} lacks notion.databases ids for: {', '.join(missing)} —"
            " run ops/bootstrap_notion.py to (re)create them"
        )
    return NotionDatabases(**{f.name: str(databases[f.name]) for f in fields(NotionDatabases)})


# -- property payload builders (Notion API property value shapes) --------------


def _clip(text: str, limit: int = _MAX_TEXT) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def title_property(text: str) -> dict[str, Any]:
    return {"title": [{"type": "text", "text": {"content": _clip(text, _MAX_TITLE)}}]}


def rich_text_property(text: str) -> dict[str, Any]:
    if not text:
        return {"rich_text": []}
    return {"rich_text": [{"type": "text", "text": {"content": _clip(text)}}]}


def select_property(name: str) -> dict[str, Any]:
    return {"select": {"name": name}}


def url_property(url: str) -> dict[str, Any]:
    return {"url": url}


def number_property(value: float) -> dict[str, Any]:
    return {"number": value}


def checkbox_property(value: bool) -> dict[str, Any]:
    return {"checkbox": value}


def relation_property(page_id: str) -> dict[str, Any]:
    return {"relation": [{"id": page_id}]}


def date_property(iso_datetime: str) -> dict[str, Any]:
    return {"date": {"start": iso_datetime}}


def directive_details(directive: Directive) -> str:
    """The refined directive rendered for the Research Ideas Directive field."""
    lines = [f"Objective: {directive.objective}"]
    if directive.universe_hint:
        lines.append(f"Universe hint: {directive.universe_hint}")
    if directive.constraints:
        lines.append(f"Constraints: {directive.constraints}")
    return "\n".join(lines)


class NotionRecorder:
    """Best-effort writer for the research-lifecycle Notion databases.

    Share one instance per process (like StateStore); the conversation core
    records directive/run-status lifecycle points, the hypothesis poller
    records hypotheses, operator actions, and completed experiments.
    ``permalink`` (thread_ts -> Slack URL) is injectable because building a
    permalink needs the Slack WebClient + channel id, which live in app.py.
    """

    def __init__(
        self,
        notion: NotionClient,
        databases: NotionDatabases,
        store: StateStore,
        permalink: Callable[[str], str | None] | None = None,
    ) -> None:
        self._notion = notion
        self._databases = databases
        self._store = store
        self._permalink = permalink

    # -- Research Ideas -----------------------------------------------------

    def record_idea(
        self,
        thread_ts: str,
        *,
        raw_idea: str,
        directive: Directive,
        universe: str | None = None,
    ) -> str | None:
        """Create (or update, when the thread re-saves) the idea page."""
        return self._guarded(
            f"record_idea for thread {thread_ts}",
            lambda: self._record_idea(thread_ts, raw_idea, directive, universe),
        )

    def record_idea_status(
        self, thread_ts: str, status: str, universe: str | None = None
    ) -> None:
        """Move the idea page's Status (researching/stopped/completed/failed/...)."""
        self._guarded(
            f"record_idea_status {status!r} for thread {thread_ts}",
            lambda: self._record_idea_status(thread_ts, status, universe),
        )

    def _record_idea(
        self, thread_ts: str, raw_idea: str, directive: Directive, universe: str | None
    ) -> str:
        properties: dict[str, Any] = {
            "Idea": title_property(directive.objective),
            "Raw Idea": rich_text_property(raw_idea),
            "Directive": rich_text_property(directive_details(directive)),
        }
        if universe:
            properties["Universe"] = rich_text_property(universe)
        page_id = self._store.get_notion_page(PAGE_KIND_IDEA, thread_ts)
        if page_id is not None:
            self._notion.update_page(page_id, properties=properties)
            return page_id
        properties["Status"] = select_property("proposed")
        properties["Thread TS"] = rich_text_property(thread_ts)
        link = self._thread_link(thread_ts)
        if link:
            properties["Thread"] = url_property(link)
        page = self._notion.create_page(
            {"type": "database_id", "database_id": self._databases.research_ideas},
            properties,
        )
        self._store.set_notion_page(PAGE_KIND_IDEA, thread_ts, page["id"])
        return page["id"]

    def _record_idea_status(
        self, thread_ts: str, status: str, universe: str | None
    ) -> None:
        page_id = self._store.get_notion_page(PAGE_KIND_IDEA, thread_ts)
        if page_id is None:
            logger.warning(
                "no Research Ideas page recorded for thread %s — cannot record"
                " status %r (was the directive saved before Notion recording"
                " was enabled?)",
                thread_ts,
                status,
            )
            return
        properties: dict[str, Any] = {"Status": select_property(status)}
        if universe:
            properties["Universe"] = rich_text_property(universe)
        self._notion.update_page(page_id, properties=properties)

    # -- Hypothesis Log -------------------------------------------------------

    def record_hypothesis(
        self, thread_ts: str, interaction_key: str, content: Mapping[str, Any]
    ) -> str | None:
        """One row per proposed hypothesis, Action 'pending', linked to the idea."""
        return self._guarded(
            f"record_hypothesis {interaction_key}",
            lambda: self._record_hypothesis(thread_ts, interaction_key, content),
        )

    def record_hypothesis_action(
        self, interaction_key: str, action: str, operator_input: str | None = None
    ) -> None:
        """Record the operator's action (approved/edited/rejected/cancelled/...)."""
        self._guarded(
            f"record_hypothesis_action {action!r} for {interaction_key}",
            lambda: self._record_hypothesis_action(interaction_key, action, operator_input),
        )

    def _record_hypothesis(
        self, thread_ts: str, interaction_key: str, content: Mapping[str, Any]
    ) -> str:
        properties: dict[str, Any] = {
            "Hypothesis": title_property(
                str(content.get("hypothesis") or "") or "(no hypothesis text)"
            ),
            "Details": rich_text_property(json.dumps(dict(content), sort_keys=True)),
            "Action": select_property("pending"),
            "Interaction Key": rich_text_property(interaction_key),
        }
        idea_page = self._store.get_notion_page(PAGE_KIND_IDEA, thread_ts)
        if idea_page is not None:
            properties["Idea"] = relation_property(idea_page)
        page = self._notion.create_page(
            {"type": "database_id", "database_id": self._databases.hypothesis_log},
            properties,
        )
        self._store.set_notion_page(PAGE_KIND_HYPOTHESIS, interaction_key, page["id"])
        return page["id"]

    def _record_hypothesis_action(
        self, interaction_key: str, action: str, operator_input: str | None
    ) -> None:
        page_id = self._store.get_notion_page(PAGE_KIND_HYPOTHESIS, interaction_key)
        if page_id is None:
            logger.warning(
                "no Hypothesis Log page recorded for interaction %s — cannot"
                " record action %r",
                interaction_key,
                action,
            )
            return
        properties: dict[str, Any] = {"Action": select_property(action)}
        if operator_input is not None:
            properties["Operator Input"] = rich_text_property(operator_input)
        self._notion.update_page(page_id, properties=properties)

    # -- Backtest Results -----------------------------------------------------

    def record_backtest(
        self,
        thread_ts: str,
        *,
        title: str,
        metrics: Mapping[str, float],
        sharpe: float | None,
        sota: bool,
        workspace_path: str,
        universe: str | None,
    ) -> str | None:
        """One row per completed experiment: headline metrics + SOTA + workspace."""
        return self._guarded(
            f"record_backtest {title!r} for thread {thread_ts}",
            lambda: self._record_backtest(
                thread_ts, title, metrics, sharpe, sota, workspace_path, universe
            ),
        )

    def _record_backtest(
        self,
        thread_ts: str,
        title: str,
        metrics: Mapping[str, float],
        sharpe: float | None,
        sota: bool,
        workspace_path: str,
        universe: str | None,
    ) -> str:
        properties: dict[str, Any] = {
            "Experiment": title_property(title),
            "SOTA": checkbox_property(sota),
            "Workspace": rich_text_property(workspace_path),
        }
        if universe:
            properties["Universe"] = rich_text_property(universe)
        # Property names in the Backtest Results schema match the display
        # labels in summary.METRIC_SPECS (both follow notion-schema.md).
        for label, keys, _style in summary.METRIC_SPECS:
            value = next((metrics[k] for k in keys if k in metrics), None)
            if value is not None and math.isfinite(value):
                properties[label] = number_property(value)
        if sharpe is not None and math.isfinite(sharpe):
            properties["Sharpe"] = number_property(sharpe)
        idea_page = self._store.get_notion_page(PAGE_KIND_IDEA, thread_ts)
        if idea_page is not None:
            properties["Idea"] = relation_property(idea_page)
        page = self._notion.create_page(
            {"type": "database_id", "database_id": self._databases.backtest_results},
            properties,
        )
        return page["id"]

    # -- Decision Log -----------------------------------------------------------

    def record_decision(
        self,
        *,
        title: str,
        decision_type: str,
        details: str,
        thread_ts: str | None = None,
    ) -> str | None:
        """One row per deliberate decision that changes what trades (US-033).

        ``decision_type`` is the schema's Type select (promotion / halt /
        resume / universe / other); ``thread_ts`` links the row to the
        thread's Research Ideas page when one is recorded.
        """
        return self._guarded(
            f"record_decision {title!r}",
            lambda: self._record_decision(title, decision_type, details, thread_ts),
        )

    def _record_decision(
        self, title: str, decision_type: str, details: str, thread_ts: str | None
    ) -> str:
        properties: dict[str, Any] = {
            "Decision": title_property(title),
            "Type": select_property(decision_type),
            "Details": rich_text_property(details),
            "Decided At": date_property(datetime.now(timezone.utc).isoformat()),
        }
        if thread_ts is not None:
            idea_page = self._store.get_notion_page(PAGE_KIND_IDEA, thread_ts)
            if idea_page is not None:
                properties["Idea"] = relation_property(idea_page)
        page = self._notion.create_page(
            {"type": "database_id", "database_id": self._databases.decision_log},
            properties,
        )
        return page["id"]

    # -- internals --------------------------------------------------------------

    def _thread_link(self, thread_ts: str) -> str | None:
        if self._permalink is None:
            return None
        try:
            return self._permalink(thread_ts)
        except Exception:  # noqa: BLE001 - a permalink is nice-to-have, never blocking
            logger.warning("could not resolve a Slack permalink for %s", thread_ts)
            return None

    def _guarded(self, action: str, fn: Callable[[], _T]) -> _T | None:
        """Run one recording step; log-and-swallow so callers never break."""
        try:
            return fn()
        except Exception:  # noqa: BLE001 - recording must never break the flow it observes
            logger.exception("Notion recording failed: %s", action)
            return None
