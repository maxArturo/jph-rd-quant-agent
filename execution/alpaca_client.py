"""Typed Alpaca paper-trading client for account, positions, and orders.

All requests are bare HTTPS: no APCA-API-KEY-ID / APCA-API-SECRET-KEY header
appears anywhere in this module (a test greps for it). The OneCLI proxy
injects both paper credentials when the process runs under
`onecli run --agent rdq-exec-paper` (the identity with the two
paper-api.alpaca.markets secret assignments).

PAPER ONLY: the base URL defaults to the paper host and the constructor
refuses the live host outright — live trading is out of scope for this repo
(PLAN.md; there is no rdq-exec-live identity).

Retry policy: only 429 is retried (Retry-After honored, exponential
fallback) — a 429 means the request was not processed. 5xx responses are
NEVER retried here: a timeout/500 on POST /v2/orders is ambiguous (the order
may have been accepted) and blind retries could double-submit.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests

BASE_URL = "https://paper-api.alpaca.markets"
_LIVE_HOST = "api.alpaca.markets"  # refused: live trading is out of scope

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 4
DEFAULT_BACKOFF_BASE_SECONDS = 1.0

ORDER_TYPES = frozenset({"market", "limit"})
ORDER_SIDES = frozenset({"buy", "sell"})


class AlpacaError(RuntimeError):
    """Base error for Alpaca client failures."""


class AlpacaAuthError(AlpacaError):
    """401/403 from Alpaca: the OneCLI proxy did not inject valid credentials."""


class AlpacaRateLimitError(AlpacaError):
    """429 from Alpaca persisted beyond the retry budget."""


@dataclass(frozen=True)
class Account:
    id: str
    status: str
    currency: str
    equity: float
    cash: float
    buying_power: float


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float  # signed: negative for short positions
    side: str  # "long" | "short"
    avg_entry_price: float
    current_price: float | None
    market_value: float | None


@dataclass(frozen=True)
class Order:
    id: str
    client_order_id: str
    symbol: str
    qty: float | None  # None for notional orders
    notional: float | None
    side: str
    order_type: str  # API field "type"
    time_in_force: str
    limit_price: float | None
    status: str
    filled_qty: float
    filled_avg_price: float | None
    submitted_at: str | None  # ISO timestamp, kept verbatim


def _float(value: Any, field: str, context: str) -> float:
    """Alpaca returns numeric fields as strings ("equity": "100000.25")."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise AlpacaError(f"unparseable {field}={value!r} in {context} response") from exc


def _opt_float(value: Any, field: str, context: str) -> float | None:
    if value is None:
        return None
    return _float(value, field, context)


def _parse_account(row: dict[str, Any]) -> Account:
    return Account(
        id=str(row["id"]),
        status=str(row.get("status", "")),
        currency=str(row.get("currency", "USD")),
        equity=_float(row["equity"], "equity", "account"),
        cash=_float(row["cash"], "cash", "account"),
        buying_power=_float(row["buying_power"], "buying_power", "account"),
    )


def _parse_position(row: dict[str, Any]) -> Position:
    return Position(
        symbol=str(row["symbol"]),
        qty=_float(row["qty"], "qty", "positions"),
        side=str(row.get("side", "")),
        avg_entry_price=_float(row["avg_entry_price"], "avg_entry_price", "positions"),
        current_price=_opt_float(row.get("current_price"), "current_price", "positions"),
        market_value=_opt_float(row.get("market_value"), "market_value", "positions"),
    )


def _parse_order(row: dict[str, Any]) -> Order:
    return Order(
        id=str(row["id"]),
        client_order_id=str(row.get("client_order_id", "")),
        symbol=str(row["symbol"]),
        qty=_opt_float(row.get("qty"), "qty", "orders"),
        notional=_opt_float(row.get("notional"), "notional", "orders"),
        side=str(row["side"]),
        order_type=str(row.get("type", "")),
        time_in_force=str(row.get("time_in_force", "")),
        limit_price=_opt_float(row.get("limit_price"), "limit_price", "orders"),
        status=str(row["status"]),
        filled_qty=_float(row.get("filled_qty", 0), "filled_qty", "orders"),
        filled_avg_price=_opt_float(row.get("filled_avg_price"), "filled_avg_price", "orders"),
        submitted_at=str(row["submitted_at"]) if row.get("submitted_at") is not None else None,
    )


class AlpacaClient:
    """Minimal typed client for the Alpaca v2 trading API through the OneCLI proxy.

    The base URL is configurable (e.g. for test stubs) but defaults to the
    paper host; the live host is refused at construction time.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        session: Any | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        host = urlparse(base_url).hostname or ""
        if host == _LIVE_HOST:
            raise ValueError(
                f"refusing live trading host {_LIVE_HOST}: this repo is paper-only "
                "(PLAN.md; no rdq-exec-live identity exists)"
            )
        self.base_url = base_url.rstrip("/")
        self.session = session if session is not None else requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self._sleep = sleep

    def get_account(self) -> Account:
        """GET /v2/account: current paper account snapshot."""
        row = self._request("GET", "/v2/account")
        return _parse_account(self._expect_dict(row, "/v2/account"))

    def get_positions(self) -> list[Position]:
        """GET /v2/positions: all open positions (empty list when flat)."""
        rows = self._expect_list(self._request("GET", "/v2/positions"), "/v2/positions")
        return [_parse_position(row) for row in rows]

    def list_orders(
        self,
        status: str = "open",
        limit: int | None = None,
        symbols: list[str] | None = None,
    ) -> list[Order]:
        """GET /v2/orders: orders filtered by status ("open" | "closed" | "all")."""
        params: dict[str, str] = {"status": status}
        if limit is not None:
            params["limit"] = str(limit)
        if symbols:
            params["symbols"] = ",".join(symbols)
        rows = self._expect_list(self._request("GET", "/v2/orders", params=params), "/v2/orders")
        return [_parse_order(row) for row in rows]

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "limit",
        time_in_force: str = "day",
        limit_price: float | None = None,
        client_order_id: str | None = None,
    ) -> Order:
        """POST /v2/orders: submit an order; returns the accepted order.

        Quantities and prices are sent as strings (the API's canonical wire
        form). Only market/limit types are supported — that is all the
        rebalancer (US-032/034) ever submits.
        """
        if side not in ORDER_SIDES:
            raise ValueError(f"side must be one of {sorted(ORDER_SIDES)}, got {side!r}")
        if order_type not in ORDER_TYPES:
            raise ValueError(f"order_type must be one of {sorted(ORDER_TYPES)}, got {order_type!r}")
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty!r}")
        if order_type == "limit" and limit_price is None:
            raise ValueError("limit orders require limit_price")
        if order_type == "market" and limit_price is not None:
            raise ValueError("market orders must not set limit_price")
        payload: dict[str, Any] = {
            "symbol": symbol,
            "qty": _decimal_str(qty),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if limit_price is not None:
            payload["limit_price"] = _decimal_str(limit_price)
        if client_order_id is not None:
            payload["client_order_id"] = client_order_id
        row = self._request("POST", "/v2/orders", json_payload=payload)
        return _parse_order(self._expect_dict(row, "/v2/orders"))

    def cancel_order(self, order_id: str) -> None:
        """DELETE /v2/orders/{id}: cancel one order (204 on success)."""
        self._request("DELETE", f"/v2/orders/{order_id}")

    def _expect_dict(self, payload: Any, path: str) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise AlpacaError(f"expected a JSON object from {path}, got: {str(payload)[:200]}")
        return payload

    def _expect_list(self, payload: Any, path: str) -> list[dict[str, Any]]:
        if not isinstance(payload, list):
            raise AlpacaError(f"expected a JSON list from {path}, got: {str(payload)[:200]}")
        return payload

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, str] | None = None,
        json_payload: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        attempt = 0
        while True:
            response = self.session.request(
                method,
                url,
                params=params,
                json=json_payload,
                timeout=self.timeout,
            )
            status = getattr(response, "status_code", None)
            if status == 429:
                if attempt >= self.max_retries:
                    raise AlpacaRateLimitError(
                        f"Alpaca kept returning 429 for {method} {path} after "
                        f"{self.max_retries} retries; back off and retry later"
                    )
                self._sleep(self._retry_delay(response, attempt))
                attempt += 1
                continue
            if status in (401, 403):
                raise AlpacaAuthError(
                    f"Alpaca returned {status} for {method} {path}: no valid paper "
                    "credentials were injected. Run this process under `onecli run "
                    "--agent rdq-exec-paper` and make sure BOTH "
                    "paper-api.alpaca.markets secrets (key id + secret key) are "
                    "vaulted and assigned (ops/setup_onecli.sh, then verify with "
                    f"ops/check_onecli.sh). Body: {_error_detail(response)}"
                )
            if status is None or not 200 <= int(status) < 300:
                raise AlpacaError(
                    f"Alpaca returned HTTP {status} for {method} {path}: "
                    f"{_error_detail(response)}"
                )
            if status == 204:
                return None
            return response.json()

    def _retry_delay(self, response: Any, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After") if response.headers else None
        if retry_after is not None:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass  # non-numeric Retry-After: fall back to exponential
        return DEFAULT_BACKOFF_BASE_SECONDS * (2**attempt)


def _decimal_str(value: float) -> str:
    """Wire form for qty/price: '15' for whole numbers, '15.5' otherwise."""
    return f"{value:g}" if value != int(value) else str(int(value))


def _error_detail(response: Any) -> str:
    """Alpaca error bodies are {'code': ..., 'message': ...}."""
    try:
        body = response.json()
    except Exception:
        body = None
    if isinstance(body, dict) and ("code" in body or "message" in body):
        return f"{body.get('code', 'unknown_code')}: {body.get('message', '')}"[:300]
    return str(getattr(response, "text", ""))[:300]
