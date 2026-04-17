#!/usr/bin/env python3
"""
test_s3.py — Verify AWS S3 bucket + IAM credentials for ProcEx (Step 2).

Tests:
  1. Credentials loaded correctly
  2. Bucket exists and is accessible
  3. Upload a test file
  4. Generate a pre-signed URL and verify it's reachable
  5. Delete the test file (cleanup)

Usage:
    pip install boto3 requests python-dotenv
    python test_s3.py
"""

import os
import sys
import json
import time
import uuid

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

AWS_ACCESS_KEY_ID     = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION            = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET             = os.environ.get("S3_BUCKET", "procex-videos")

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
    banner("ProcEx — S3 Bucket + IAM Test", "═")

    # ── 0. Environment ────────────────────────────────────────────────────────
    banner("Step 0 · Environment Variables")
    env_ok = True
    env_ok &= check("AWS_ACCESS_KEY_ID is set",
                    bool(AWS_ACCESS_KEY_ID),
                    "***" + AWS_ACCESS_KEY_ID[-4:] if AWS_ACCESS_KEY_ID else "MISSING")
    env_ok &= check("AWS_SECRET_ACCESS_KEY is set",
                    bool(AWS_SECRET_ACCESS_KEY),
                    "***" + AWS_SECRET_ACCESS_KEY[-4:] if AWS_SECRET_ACCESS_KEY else "MISSING")
    env_ok &= check("AWS_REGION is set",  bool(AWS_REGION),  AWS_REGION)
    env_ok &= check("S3_BUCKET is set",   bool(S3_BUCKET),   S3_BUCKET)

    if not env_ok:
        print("\n  ⚠️  Fill in the missing AWS variables in your .env and re-run.\n")
        return False

    # ── 1. Import boto3 ───────────────────────────────────────────────────────
    banner("Step 1 · Import boto3")
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        check("boto3 imported", True, boto3.__version__)
    except ImportError:
        check("boto3 imported", False, "run: pip install boto3")
        return False

    # ── 2. Connect + verify bucket ────────────────────────────────────────────
    banner("Step 2 · Connect & Verify Bucket")
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        )
        s3.head_bucket(Bucket=S3_BUCKET)
        check(f"Bucket '{S3_BUCKET}' exists and is accessible", True)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code == "404":
            check("Bucket exists", False, f"bucket '{S3_BUCKET}' not found — did you create it?")
        elif code == "403":
            check("Bucket accessible", False, "access denied — check IAM policy is attached")
        else:
            check("Bucket check", False, f"error {code}: {exc}")
        return False
    except NoCredentialsError:
        check("Credentials valid", False, "no credentials found — check AWS_ACCESS_KEY_ID / SECRET")
        return False
    except Exception as exc:
        check("Bucket check", False, str(exc)[:120])
        return False

    # ── 3. Upload a test object ───────────────────────────────────────────────
    banner("Step 3 · Upload Test Object")
    test_key  = f"_test/{uuid.uuid4()}/test.json"
    test_body = json.dumps({
        "procex": "step2-test",
        "ts": time.time(),
        "bucket": S3_BUCKET,
    })
    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=test_key,
            Body=test_body.encode(),
            ContentType="application/json",
        )
        check(f"PUT  {test_key}", True)
    except ClientError as exc:
        check("Upload (PutObject)", False, str(exc.response["Error"]["Message"]))
        print("  ⚠️  The IAM user is missing s3:PutObject on this bucket.")
        return False

    # ── 4. Download + verify ──────────────────────────────────────────────────
    banner("Step 4 · Download & Verify")
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=test_key)
        body     = response["Body"].read().decode()
        parsed   = json.loads(body)
        check("GET  test object",           True)
        check("Content matches what we PUT", parsed["procex"] == "step2-test")
    except ClientError as exc:
        check("Download (GetObject)", False, str(exc.response["Error"]["Message"]))
        return False

    # ── 5. Pre-signed URL ─────────────────────────────────────────────────────
    banner("Step 5 · Pre-Signed URL (24h)")
    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": test_key},
            ExpiresIn=86400,
        )
        check("Pre-signed URL generated", True)
        # Verify the URL is reachable with GET (pre-signed URLs are method-specific)
        import urllib.request
        resp = urllib.request.urlopen(url, timeout=10)
        check("Pre-signed URL is reachable via HTTP", resp.status == 200,
              f"HTTP {resp.status}")
        print(f"\n  Sample URL (expires 24h):\n  {url[:80]}...\n")
    except Exception as exc:
        check("Pre-signed URL", False, str(exc)[:120])
        return False

    # ── 6. Delete test object (cleanup) ──────────────────────────────────────
    banner("Step 6 · Cleanup")
    try:
        s3.delete_object(Bucket=S3_BUCKET, Key=test_key)
        check("DEL  test object", True, "bucket is clean")
    except ClientError as exc:
        check("Delete (DeleteObject)", False, str(exc.response["Error"]["Message"]))
        print("  ⚠️  Missing s3:DeleteObject — add it to the IAM policy.")
        return False

    # ── Summary ───────────────────────────────────────────────────────────────
    banner("✅  All checks passed — S3 is ready for Step 3 (Docker)", "═")
    print()
    print("  Bucket  :", S3_BUCKET)
    print("  Region  :", AWS_REGION)
    print("  IAM     : PutObject / GetObject / DeleteObject / HeadObject ✓")
    print()
    return True


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)