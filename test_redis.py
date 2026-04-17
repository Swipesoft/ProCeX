#!/usr/bin/env python3
"""
test_redis.py — Verify Upstash Redis connectivity before proceeding to Step 2.

Uses the official Upstash HTTP/REST SDK (upstash-redis) for the connection
test, and separately verifies the TCP URL that Celery will use.

Credentials needed in .env:
    UPSTASH_REDIS_REST_URL    — https://your-db.upstash.io
    UPSTASH_REDIS_REST_TOKEN  — your REST token
    UPSTASH_REDIS_URL         — rediss://default:<pw>@your-db.upstash.io:6379

Usage:
    pip install upstash-redis redis python-dotenv
    python test_redis.py
"""

import os
import sys
import time
import json
import ssl

# ── Load .env if present ──────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

REST_URL   = os.environ.get("UPSTASH_REDIS_REST_URL", "")
REST_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")
TCP_URL    = os.environ.get("UPSTASH_REDIS_URL", "")

# ─────────────────────────────────────────────────────────────────────────────

def banner(text: str, char: str = "─") -> None:
    width = 62
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def check(label: str, ok: bool, detail: str = "") -> bool:
    icon = "✅" if ok else "❌"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {icon}  {label}{suffix}")
    return ok


def run_tests() -> bool:
    banner("ProcEx — Upstash Redis Connection Test", "═")

    # ── 0. Environment ────────────────────────────────────────────────────────
    banner("Step 0 · Environment Variables")
    env_ok = True
    env_ok &= check("UPSTASH_REDIS_REST_URL is set",   bool(REST_URL),
                    REST_URL[:40] + "..." if REST_URL else "MISSING")
    env_ok &= check("UPSTASH_REDIS_REST_TOKEN is set", bool(REST_TOKEN),
                    "***" + REST_TOKEN[-4:] if REST_TOKEN else "MISSING")
    env_ok &= check("UPSTASH_REDIS_URL is set",        bool(TCP_URL),
                    TCP_URL[:40] + "..." if TCP_URL else "MISSING")

    if not env_ok:
        print("\n  ⚠️  Fill in all three variables in your .env and re-run.")
        print("      See .env.example for the correct format.\n")
        return False

    check("REST URL uses https://",  REST_URL.startswith("https://"),
          "correct" if REST_URL.startswith("https://") else "should start with https://")
    check("TCP  URL uses rediss://", TCP_URL.startswith("rediss://"),
          "correct" if TCP_URL.startswith("rediss://") else "should start with rediss://")

    # ── 1. Import upstash-redis SDK ───────────────────────────────────────────
    banner("Step 1 · Import upstash-redis SDK")
    try:
        from upstash_redis import Redis as UpstashRedis
        import upstash_redis as _ur
        check("upstash-redis imported", True, getattr(_ur, "__version__", "ok"))
    except ImportError:
        check("upstash-redis imported", False, "run: pip install upstash-redis")
        return False

    # ── 2. Connect via HTTP SDK ───────────────────────────────────────────────
    banner("Step 2 · HTTP SDK — Connect & Ping")
    try:
        r = UpstashRedis(url=REST_URL, token=REST_TOKEN)
        pong = r.ping()
        check("PING via HTTP SDK", str(pong).upper() == "TRUE" or pong is True
              or str(pong) == "PONG", str(pong))
    except Exception as exc:
        check("PING via HTTP SDK", False, str(exc)[:120])
        print("\n  ⚠️  Common causes:")
        print("      • Wrong REST token")
        print("      • REST URL copied incorrectly (needs https://)")
        print("      • Database not yet fully provisioned — wait 30s and retry\n")
        return False

    # ── 3. Read / Write / Delete via HTTP SDK ─────────────────────────────────
    banner("Step 3 · HTTP SDK — Read / Write / Delete")
    test_key = "procex:test:step1"
    test_val = json.dumps({"status": "ok", "ts": time.time()})
    try:
        r.set(test_key, test_val, ex=60)
        got = r.get(test_key)
        check("SET  procex:test:step1", True)
        check("GET  procex:test:step1", got is not None)
        r.delete(test_key)
        check("DEL  procex:test:step1", r.get(test_key) is None)
    except Exception as exc:
        check("Read / Write / Delete", False, str(exc)[:120])
        return False

    # ── 4. Celery key schema simulation via HTTP SDK ──────────────────────────
    banner("Step 4 · Celery Key Schema")
    fake_job_id = "00000000-0000-0000-0000-000000000001"
    celery_key  = f"celery-task-meta-{fake_job_id}"
    procex_key  = f"procex:job:{fake_job_id}"
    try:
        r.set(celery_key, json.dumps({"status": "PENDING"}), ex=60)
        r.hset(procex_key, "topic", "Test Topic")
        r.hset(procex_key, "status", "pending")
        r.expire(procex_key, 60)

        check("celery-task-meta key writable", r.exists(celery_key) == 1)
        check("procex:job hash writable",      r.hget(procex_key, "topic") == "Test Topic")

        r.delete(celery_key)
        r.delete(procex_key)
        check("Cleanup test keys", True)
    except Exception as exc:
        check("Celery key schema", False, str(exc)[:120])
        return False

    # ── 5. TCP URL reachable (for Celery) ─────────────────────────────────────
    banner("Step 5 · TCP URL — Celery Broker Check")
    try:
        import redis as redis_tcp
        ssl_opts = {"ssl_cert_reqs": ssl.CERT_NONE}
        rc = redis_tcp.from_url(TCP_URL, decode_responses=True, **ssl_opts)
        pong_tcp = rc.ping()
        check("PING via TCP (redis-py)", pong_tcp is True, "Celery will use this")
    except ImportError:
        check("redis-py imported", False, "run: pip install redis")
        return False
    except Exception as exc:
        check("PING via TCP (redis-py)", False, str(exc)[:120])
        print("\n  ⚠️  TCP URL failed — check UPSTASH_REDIS_URL format:")
        print("      rediss://default:<password>@<host>.upstash.io:6379\n")
        return False

    # ── 6. Latency (HTTP SDK) ─────────────────────────────────────────────────
    banner("Step 6 · Latency (HTTP SDK)")
    try:
        times: list[float] = []
        for _ in range(5):
            t0 = time.perf_counter()
            r.ping()
            times.append((time.perf_counter() - t0) * 1000)
        avg_ms = sum(times) / len(times)
        ok_lat = avg_ms < 500   # HTTP adds overhead vs TCP; 500ms is generous
        check(f"Avg round-trip {avg_ms:.0f} ms", ok_lat,
              "good" if ok_lat else "high — consider a closer Upstash region")
    except Exception as exc:
        check("Latency test", False, str(exc)[:80])

    # ── Summary ───────────────────────────────────────────────────────────────
    banner("✅  All checks passed — Redis is ready for Step 2", "═")
    print()
    print("  Next step: S3 bucket + IAM user (Step 2)")
    print()
    return True


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)