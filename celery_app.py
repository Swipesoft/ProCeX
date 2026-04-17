"""
celery_app.py — ProcEx Celery application

Broker + backend: Upstash Redis (TLS)
Branch: gemma-mode

Upstash Redis URLs start with  rediss://  (double-s) which enables TLS.
The SSL broker/backend transport options below disable certificate hostname
verification — required for Upstash's shared TLS endpoint.
"""

import os
import ssl
from celery import Celery

# ── Redis URL ────────────────────────────────────────────────────────────────
# Upstash gives you:  rediss://default:<password>@<host>.upstash.io:6379
# Local Docker dev:   redis://localhost:6379
REDIS_URL: str = os.environ["REDIS_URL"]

# ── Celery application ───────────────────────────────────────────────────────
app = Celery(
    "procex",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks"],          # auto-discover tasks module
)

# ── SSL options (needed only for Upstash's rediss:// URLs) ───────────────────
_is_tls = REDIS_URL.startswith("rediss://")
_ssl_opts: dict = (
    {
        "ssl_cert_reqs": ssl.CERT_NONE,   # Upstash uses shared cert; skip hostname check
    }
    if _is_tls
    else {}
)

# ── Core configuration ───────────────────────────────────────────────────────
app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Reliability
    task_track_started=True,
    task_acks_late=True,              # re-queue task if worker dies mid-run
    worker_prefetch_multiplier=1,     # one task per worker process at a time

    # Time limits  (ProcEx pipeline ≈ 5–20 min depending on video length)
    task_time_limit=35 * 60,          # 35 min hard kill
    task_soft_time_limit=30 * 60,     # 30 min graceful raise SoftTimeLimitExceeded

    # Result TTL
    result_expires=86_400,            # purge results after 24 h

    # Queue
    task_default_queue="procex",
    task_queues={
        "procex":         {"exchange": "procex",         "routing_key": "procex"},
        "procex-premium": {"exchange": "procex-premium", "routing_key": "procex-premium"},
    },

    # Redis / Upstash transport options
    broker_transport_options={
        "visibility_timeout": 36 * 60,  # must be > task_time_limit
        **(_ssl_opts),
    },
    redis_backend_use_ssl=_ssl_opts if _is_tls else None,

    # Logging
    worker_hijack_root_logger=False,
)

# ── Convenience: allow running  python celery_app.py  for quick smoke test ───
if __name__ == "__main__":
    print("Celery app configured.")
    print(f"  Broker  : {REDIS_URL[:30]}...")
    print(f"  TLS     : {_is_tls}")
    print(f"  Queues  : {list(app.conf.task_queues.keys())}")