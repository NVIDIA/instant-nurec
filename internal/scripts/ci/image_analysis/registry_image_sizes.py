#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Query nvcr.io container image sizes from registry without pulling (Registry V2 API).
Auth from ~/.docker/config.json (auths["nvcr.io"].auth = base64("$oauthtoken:<api_key>")).
"""

import argparse
import base64
import json
import os
import re
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request


REGISTRY = "nvcr.io"
REGISTRY_BASE = f"https://{REGISTRY}"
ACCEPT = "application/vnd.docker.distribution.manifest.v2+json"


def nvcr_credentials():
    """Return (username, password) from Docker config auths for nvcr.io. Exits if missing."""
    path = os.path.join(os.path.expanduser("~/.docker"), "config.json")
    if not os.path.isfile(path):
        print("registry_image_sizes: Docker config not found", file=sys.stderr)
        sys.exit(1)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        data = {}
    auths = data.get("auths") or {}
    entry = auths.get(REGISTRY) or {}
    b64 = entry.get("auth")
    if not b64:
        print(f"registry_image_sizes: no auth for {REGISTRY} in {path}", file=sys.stderr)
        sys.exit(1)
    try:
        decoded = base64.b64decode(b64, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        print(f"registry_image_sizes: invalid auth for {REGISTRY} in {path}", file=sys.stderr)
        sys.exit(1)
    if ":" in decoded:
        u, _, p = decoded.partition(":")
        return (u.strip(), p.strip())
    return (decoded.strip(), "")


def parse_www_authenticate(header: str):
    s = header.strip()
    if s.lower().startswith("bearer "):
        s = s[7:].strip()
    out = {}
    for part in re.split(r",\s*", s):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip().lower()] = v.strip().strip('"')
    return out


def get_token(realm: str, service: str, scope: str, user: str, password: str):
    url = realm + ("&" if "?" in realm else "?") + urllib.parse.urlencode({"service": service, "scope": scope})
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode())
    with urllib.request.urlopen(req, context=ssl.create_default_context(), timeout=30) as r:
        data = json.loads(r.read().decode())
    return data.get("token") or data.get("access_token") or ""


def registry_request(url: str, token=None, accept=None):
    req = urllib.request.Request(url, method="GET")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if accept:
        req.add_header("Accept", accept)
    with urllib.request.urlopen(req, context=ssl.create_default_context(), timeout=60) as r:
        return json.loads(r.read().decode())


def obtain_token(repository: str, user: str, password: str) -> str:
    url = f"{REGISTRY_BASE}/v2/"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, method="GET"), context=ssl.create_default_context(), timeout=15
        ):
            return ""
    except urllib.error.HTTPError as e:
        if e.code != 401:
            raise
        auth = e.headers.get("Www-Authenticate")
        if not auth or "Bearer" not in auth:
            raise RuntimeError("registry did not return Bearer challenge")
        p = parse_www_authenticate(auth)
        realm = p.get("realm")
        if not realm:
            raise RuntimeError("no realm in Www-Authenticate")
        scope = (p.get("scope") or "").strip() or f"repository:{repository}:pull"
        return get_token(realm, p.get("service", "container_registry"), scope, user, password)


def fetch_manifest(repository: str, reference: str, token: str):
    url = f"{REGISTRY_BASE}/v2/{repository}/manifests/{reference}"
    raw = registry_request(url, token=token, accept=ACCEPT)
    if not isinstance(raw, dict) or "manifests" not in raw:
        return raw
    for m in raw.get("manifests", []):
        if not isinstance(m, dict):
            continue
        p = m.get("platform") or {}
        if p.get("architecture") == "amd64" and p.get("os") == "linux":
            d = m.get("digest")
            if d:
                return registry_request(f"{REGISTRY_BASE}/v2/{repository}/manifests/{d}", token=token, accept=ACCEPT)
            break
    m0 = raw.get("manifests", [None])[0]
    if isinstance(m0, dict) and m0.get("digest"):
        return registry_request(f"{REGISTRY_BASE}/v2/{repository}/manifests/{m0['digest']}", token=token, accept=ACCEPT)
    return raw


def size_from_manifest(manifest: dict) -> int:
    total = manifest.get("config", {}).get("size", 0) if isinstance(manifest.get("config"), dict) else 0
    for layer in manifest.get("layers", []):
        total += layer.get("size", 0)
    return total


def format_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GiB"
    if n >= 1024**2:
        return f"{n / 1024**2:.2f} MiB"
    if n >= 1024:
        return f"{n / 1024:.2f} KiB"
    return f"{n} B"


def main():
    p = argparse.ArgumentParser(description="Print nvcr.io image sizes (compressed) from registry.")
    p.add_argument("--images", required=True, metavar="LIST", help="Comma-separated nvcr.io/repo:tag list")
    p.add_argument("--json", action="store_true", help="Output JSON {image: size_bytes}")
    args = p.parse_args()
    images = [s.strip() for s in args.images.split(",") if s.strip()]
    if not images:
        p.error("--images required")

    for img in images:
        if ":" not in img:
            p.error("each image must be nvcr.io/repo:tag")
        rest, _ = img.rsplit(":", 1)
        if not rest.startswith("nvcr.io/"):
            p.error(f"only nvcr.io images allowed, got: {img}")

    user, password = nvcr_credentials()
    if not password:
        print("registry_image_sizes: nvcr.io auth must be base64($oauthtoken:<api_key>)", file=sys.stderr)
        sys.exit(1)
    sizes: dict[str, int] = {}
    failed = 0
    for img in images:
        rest, ref = img.rsplit(":", 1)
        repo = rest.split("/", 1)[1]
        try:
            token = obtain_token(repo, user, password)
            manifest = fetch_manifest(repo, ref, token)
            sizes[img] = size_from_manifest(manifest)
        except Exception as e:
            print(f"  {img}: {e}", file=sys.stderr)
            failed += 1

    if args.json:
        print(json.dumps(sizes, indent=2))
        return 1 if failed else 0

    print("Container image sizes (compressed, from registry manifest):")
    print("------------------------------------------------------------")
    for img in images:
        if img in sizes:
            print(f"  {img}: {format_size(sizes[img])}")
    print("------------------------------------------------------------")
    if sizes:
        print(f"  Total: {format_size(sum(sizes.values()))}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
