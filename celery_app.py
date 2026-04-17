"""
celery_app.py — ProcEx Celery application
Branch: gemma-mode

Celery requires a persistent TCP connection to Redis — it cannot use the
Upstash HTTP/REST SDK. We use the standard redis-py client under the hood
via the  UPSTASH_REDIS_URL  (rediss://) credential from the Upstash console.

Upstash console gives you TWO sets of credentials:
  • UPSTASH_REDIS_REST_URL + UPSTASH_REDIS_REST_TOKEN  → HTTP SDK (upstash-redis)
  • UPSTASH_REDIS_URL                                  → TCP client (this file)

This file uses UPSTASH_REDIS_URL.
"""

import os
import ssl
from celery import Celery

# ── Redis URL (TCP) ───────────────────────────────────────────────────────────
# From Upstash console → your database → "Connect" → "Redis Clients" tab
# Format:  rediss://default:<password>@<host>.upstash.io:6379
UPSTASH_REDIS_URL: str = os.environ["UPSTASH_REDIS_URL"]

# ── Celery application ────────────────────────────────────────────────────────
app = Celery(
    "procex",
    broker=UPSTASH_REDIS_URL,
    backend=UPSTASH_REDIS_URL,
    include=["tasks"],
)

# ── TLS options for Upstash's shared certificate ──────────────────────────────
_is_tls = UPSTASH_REDIS_URL.startswith("rediss://")
_ssl_opts: dict = {"ssl_cert_reqs": ssl.CERT_NONE} if _is_tls else {}

# ── Core configuration ────────────────────────────────────────────────────────
app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Reliability
    task_track_started=True,
    task_acks_late=True,               # re-queue if worker dies mid-run
    worker_prefetch_multiplier=1,      # one task per worker at a time

    # Time limits  (ProcEx pipeline ≈ 5–20 min depending on video length)
    task_time_limit=35 * 60,           # 35 min hard kill
    task_soft_time_limit=30 * 60,      # 30 min graceful SoftTimeLimitExceeded

    # Result TTL
    result_expires=86_400,             # purge results after 24 h

    # Queues
    task_default_queue="procex",
    task_queues={
        "procex":         {"exchange": "procex",         "routing_key": "procex"},
        "procex-premium": {"exchange": "procex-premium", "routing_key": "procex-premium"},
    },

    # visibility_timeout must exceed task_time_limit so a slow job isn't re-queued
    broker_transport_options={
        "visibility_timeout": 40 * 60,
        **_ssl_opts,
    },
    redis_backend_use_ssl=_ssl_opts if _is_tls else None,

    worker_hijack_root_logger=False,
)

if __name__ == "__main__":
    print("Celery app configured.")
    print(f"  Broker  : {UPSTASH_REDIS_URL[:40]}...")
    print(f"  TLS     : {_is_tls}")
    print(f"  Queues  : {list(app.conf.task_queues.keys())}")