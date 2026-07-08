# orchestrator/ — module notes

- Config pattern: all env/config loading goes through `orchestrator/config.py`
  (process environment overrides the repo-root `.env`; raise `ConfigError`
  naming the missing variable and where to set it). Extend that module for new
  settings instead of reading `os.environ` elsewhere.
- Slack tokens come from the repo-root `.env` (SLACK_OAUTH_TOKEN xoxb-,
  SLACK_SOCKET_TOKEN xapp-, SLACK_CHANNEL_ID). Never route Slack through the
  OneCLI proxy and never vault these (docs/decisions.md 2026-07-08).
- Persistent state goes through `orchestrator/state.py` (`StateStore`), not ad
  hoc sqlite3 calls. It opens a short-lived connection per method, so one
  instance is safe to share between Bolt handlers and background pollers —
  never cache a `sqlite3.Connection` across threads. Extend the schema by
  adding `CREATE ... IF NOT EXISTS` statements to `_SCHEMA` (migration reruns
  on every startup). Dedup/uniqueness lives in the schema (runs.thread_ts PK
  → `DuplicateRunError`; pending_interactions.interaction_key UNIQUE → insert
  returns `None`), so restarts can't double-post.

## Testing Bolt apps (see tests/test_slack_app.py)

- Bolt >=1.15 constructs a NEW real `WebClient` per request in
  `App._init_context` — injecting a mocked client into `App(client=...)` is
  NOT enough; `say()`/`context.client` would hit the network. Also
  monkeypatch `slack_bolt.app.app.WebClient` to return the mock, but only
  AFTER `App()` is constructed (its constructor isinstance-checks that same
  symbol).
- `MagicMock(spec=WebClient)` misses instance attributes `_init_context`
  reads (`base_url`, `timeout`, `ssl`, `proxy`, `headers`, `logger`,
  `retry_handlers`) — set them explicitly on the mock.
- Pass `process_before_response=True` in tests so listeners run synchronously
  inside `App.dispatch()`; otherwise assertions race Bolt's worker threads.
  Do NOT enable it in production once handlers are slow (Claude calls):
  Slack retries events not acked within ~3s.
- Dispatch events as
  `BoltRequest(body=json.dumps({"type": "event_callback", "event": {...}, ...}), mode="socket_mode")`.
- Handlers must ignore `subtype` messages (message_changed, channel_join, ...)
  and anything with `bot_id`, or the bot replies to its own replies (loop).
  In-thread reply target: `event.get("thread_ts") or event["ts"]`.
