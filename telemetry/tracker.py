#!/usr/bin/env python3
"""Transparent tracking gateway for llama-server.

Intercepts requests on the public port, forwards them to the backend llama-server,
streams responses with zero latency, and logs detailed token usage, model/quant
metadata, and performance metrics into a local SQLite database.
"""
import argparse
import datetime
import http.server
import json
import os
import queue
import re
import socketserver
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_DEFAULT = ROOT / "telemetry" / "usage.db"
VRAM_PATH = Path("/sys/class/drm/card1/device/mem_info_vram_used")
GTT_PATH = Path("/sys/class/drm/card1/device/mem_info_gtt_used")
GiB = 2**30

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,
    timestamp_utc TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    method TEXT NOT NULL,
    status_code INTEGER,
    
    -- Model & Hardware specs
    model_alias TEXT,
    model_path TEXT,
    weight_quant TEXT,
    kv_quant TEXT,
    n_ctx INTEGER,
    engine_build TEXT,
    vram_used_gib REAL,
    gtt_used_gib REAL,
    
    -- Tokens & Content
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    reasoning_tokens INTEGER,
    cached_tokens INTEGER,
    total_tokens INTEGER,
    finish_reason TEXT,
    
    -- Performance & Timing
    duration_ms REAL,
    ttft_ms REAL,
    prompt_tok_per_sec REAL,
    generation_tok_per_sec REAL,
    
    -- Previews & Flags
    is_stream INTEGER,
    prompt_preview TEXT,
    response_preview TEXT
);

CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model_alias);
CREATE INDEX IF NOT EXISTS idx_requests_endpoint ON requests(endpoint);
"""


class DBLogger(threading.Thread):
    """Background worker for non-blocking SQLite writes."""

    def __init__(self, db_path):
        super().__init__(daemon=True)
        self.db_path = str(db_path)
        self.q = queue.Queue()
        self.init_db()

    def init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA)

    def log(self, record):
        self.q.put(record)

    def run(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        
        while True:
            record = self.q.get()
            if record is None:
                break
            try:
                keys = list(record.keys())
                placeholders = ", ".join("?" for _ in keys)
                col_names = ", ".join(keys)
                sql = f"INSERT OR REPLACE INTO requests ({col_names}) VALUES ({placeholders})"
                conn.execute(sql, list(record.values()))
                conn.commit()
            except Exception as e:
                sys.stderr.write(f"[tracker-db] write error: {e}\n")
            finally:
                self.q.task_done()


class TrackerHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        # Suppress noisy default logging
        pass

    def get_hardware_info(self):
        vram = gtt = None
        try:
            if VRAM_PATH.exists():
                vram = round(int(VRAM_PATH.read_text()) / GiB, 3)
            if GTT_PATH.exists():
                gtt = round(int(GTT_PATH.read_text()) / GiB, 3)
        except Exception:
            pass
        return vram, gtt

    def get_backend_props(self):
        now = time.time()
        # Cache backend props for 30 seconds
        if hasattr(self.server, "_cached_props") and now - getattr(self.server, "_props_time", 0) < 30:
            return self.server._cached_props

        backend_url = f"http://127.0.0.1:{self.server.backend_port}/props"
        try:
            req = urllib.request.Request(backend_url, headers={"User-Agent": "llama-tracker"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode())
                self.server._cached_props = data
                self.server._props_time = now
                return data
        except Exception:
            return getattr(self.server, "_cached_props", {})

    def forward_request(self, body=None):
        backend_url = f"http://127.0.0.1:{self.server.backend_port}{self.path}"
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length")}
        headers["Host"] = f"127.0.0.1:{self.server.backend_port}"
        
        req = urllib.request.Request(backend_url, data=body, headers=headers, method=self.command)
        return urllib.request.urlopen(req, timeout=1800)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/v1/models", "/models"):
            try:
                resp = self.forward_request()
                data = json.loads(resp.read().decode("utf-8"))
                for m in data.get("models", []):
                    caps = m.get("capabilities", [])
                    for c in ("thinking", "tools"):
                        if c not in caps:
                            caps.append(c)
                    m["capabilities"] = caps
                for m in data.get("models", []):
                    if "bonsai" in m.get("name", "").lower() or "bonsai" in m.get("model", "").lower():
                        m["name"] = "bonsai-27b"
                        m["model"] = "bonsai-27b"
                for d in data.get("data", []):
                    tags = d.get("tags", [])
                    for t in ("thinking", "tools"):
                        if t not in tags:
                            tags.append(t)
                    d["tags"] = tags
                    if "bonsai" in d.get("id", "").lower():
                        d["id"] = "bonsai-27b"
                        if "aliases" in d and "bonsai-27b" not in d["aliases"]:
                            d["aliases"].insert(0, "bonsai-27b")
                resp_bytes = json.dumps(data).encode("utf-8")
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in ("content-length", "transfer-encoding"):
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.end_headers()
                self.wfile.write(resp_bytes)
                return
            except Exception:
                pass
        elif path == "/props":
            try:
                resp = self.forward_request()
                data = json.loads(resp.read().decode("utf-8"))
                if "default_generation_settings" in data:
                    params = data["default_generation_settings"].get("params", {})
                    params["reasoning_format"] = "deepseek"
                    data["default_generation_settings"]["params"] = params
                if "chat_template_caps" in data:
                    data["chat_template_caps"]["supports_thinking"] = True
                data["model_alias"] = "bonsai-27b"
                resp_bytes = json.dumps(data).encode("utf-8")
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in ("content-length", "transfer-encoding"):
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.end_headers()
                self.wfile.write(resp_bytes)
                return
            except Exception:
                pass
        self.handle_passthrough()

    def do_POST(self):
        path = self.path.split("?")[0]
        if path not in ("/v1/chat/completions", "/chat/completions", "/v1/completions", "/completion"):
            self.handle_passthrough()
            return

        t0 = time.perf_counter()
        req_id = f"req-{int(time.time()*1000)}-{os.urandom(3).hex()}"
        timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # Read incoming body
        content_length = int(self.headers.get("Content-Length", 0))
        req_body_bytes = self.rfile.read(content_length) if content_length > 0 else b""
        
        is_stream = False
        prompt_preview = ""
        try:
            req_json = json.loads(req_body_bytes.decode("utf-8"))
            is_stream = bool(req_json.get("stream", False))
            messages = req_json.get("messages", [])
            if messages and isinstance(messages, list):
                last_msg = messages[-1].get("content", "")
                prompt_preview = str(last_msg)[:150].strip()
            elif "prompt" in req_json:
                prompt_preview = str(req_json["prompt"])[:150].strip()
        except Exception:
            pass

        # Query backend props & hardware
        props = self.get_backend_props()
        vram_gib, gtt_gib = self.get_hardware_info()
        model_alias = props.get("model_alias") or "unknown"
        model_path = props.get("model_path") or ""
        weight_quant = props.get("model_ftype") or ""
        n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx") or 0
        engine_build = props.get("build_info") or ""
        kv_quant = os.environ.get("KVTYPE") or "q4_0"

        # Forward request to backend
        try:
            resp = self.forward_request(body=req_body_bytes)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() not in ("transfer-encoding", "content-length"):
                    self.send_header(k, v)
            err_body = e.read()
            self.send_header("Content-Length", str(len(err_body)))
            self.end_headers()
            self.wfile.write(err_body)
            return
        except Exception as e:
            self.send_error(502, f"Bad Gateway: {e}")
            return

        status_code = resp.status
        self.send_response(status_code)
        for k, v in resp.headers.items():
            if k.lower() not in ("transfer-encoding", "content-length"):
                self.send_header(k, v)
        
        if is_stream:
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.handle_streaming_response(
                resp, t0, req_id, timestamp_utc, path,
                model_alias, model_path, weight_quant, kv_quant, n_ctx, engine_build,
                vram_gib, gtt_gib, prompt_preview
            )
        else:
            resp_bytes = resp.read()
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)
            self.handle_json_response(
                resp_bytes, t0, req_id, timestamp_utc, path, status_code,
                model_alias, model_path, weight_quant, kv_quant, n_ctx, engine_build,
                vram_gib, gtt_gib, prompt_preview
            )

    def handle_json_response(self, resp_bytes, t0, req_id, timestamp_utc, path, status_code,
                             model_alias, model_path, weight_quant, kv_quant, n_ctx, engine_build,
                             vram_gib, gtt_gib, prompt_preview):
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        prompt_tokens = completion_tokens = reasoning_tokens = cached_tokens = total_tokens = 0
        prompt_t_s = gen_t_s = ttft_ms = None
        finish_reason = None
        response_preview = ""

        try:
            res_json = json.loads(resp_bytes.decode("utf-8"))
            usage = res_json.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens") or 0
            completion_tokens = usage.get("completion_tokens") or 0
            total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)
            
            # Extract reasoning content if present
            choices = res_json.get("choices") or []
            if choices:
                c0 = choices[0]
                finish_reason = c0.get("finish_reason")
                msg = c0.get("message") or {}
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""
                response_preview = (content if content else reasoning)[:150].strip()
                if reasoning:
                    reasoning_tokens = len(reasoning.split())  # approximate token count if not in usage

            timings = res_json.get("timings") or {}
            prompt_t_s = timings.get("prompt_per_second")
            gen_t_s = timings.get("predicted_per_second")
            if timings.get("prompt_ms"):
                ttft_ms = round(float(timings["prompt_ms"]), 2)
            else:
                ttft_ms = duration_ms
        except Exception:
            pass

        record = {
            "id": req_id,
            "timestamp_utc": timestamp_utc,
            "endpoint": path,
            "method": "POST",
            "status_code": status_code,
            "model_alias": model_alias,
            "model_path": model_path,
            "weight_quant": weight_quant,
            "kv_quant": kv_quant,
            "n_ctx": n_ctx,
            "engine_build": engine_build,
            "vram_used_gib": vram_gib,
            "gtt_used_gib": gtt_gib,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached_tokens": cached_tokens,
            "total_tokens": total_tokens,
            "finish_reason": finish_reason,
            "duration_ms": duration_ms,
            "ttft_ms": ttft_ms,
            "prompt_tok_per_sec": prompt_t_s,
            "generation_tok_per_sec": gen_t_s,
            "is_stream": 0,
            "prompt_preview": prompt_preview,
            "response_preview": response_preview,
        }
        self.server.db_logger.log(record)

    def handle_streaming_response(self, resp, t0, req_id, timestamp_utc, path,
                                  model_alias, model_path, weight_quant, kv_quant, n_ctx, engine_build,
                                  vram_gib, gtt_gib, prompt_preview):
        ttft_ms = None
        prompt_tokens = completion_tokens = reasoning_tokens = total_tokens = 0
        prompt_t_s = gen_t_s = None
        finish_reason = None
        accumulated_content = []
        accumulated_reasoning = []

        try:
            while True:
                line = resp.readline()
                if not line:
                    break
                
                # Write HTTP chunk to client immediately
                chunk_len = f"{len(line):X}\r\n".encode()
                self.wfile.write(chunk_len + line + b"\r\n")
                self.wfile.flush()

                # Process SSE line
                line_str = line.decode("utf-8", errors="ignore").strip()
                if line_str.startswith("data: ") and line_str != "data: [DONE]":
                    try:
                        chunk_json = json.loads(line_str[6:])
                        choices = chunk_json.get("choices") or []
                        if choices:
                            c0 = choices[0]
                            delta = c0.get("delta") or {}
                            content_piece = delta.get("content") or ""
                            reasoning_piece = delta.get("reasoning_content") or ""
                            
                            if (content_piece or reasoning_piece) and ttft_ms is None:
                                ttft_ms = round((time.perf_counter() - t0) * 1000, 2)
                            
                            if content_piece:
                                accumulated_content.append(content_piece)
                            if reasoning_piece:
                                accumulated_reasoning.append(reasoning_piece)
                            
                            if c0.get("finish_reason"):
                                finish_reason = c0["finish_reason"]

                        usage = chunk_json.get("usage")
                        if usage:
                            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                            completion_tokens = usage.get("completion_tokens", completion_tokens)
                            total_tokens = usage.get("total_tokens", total_tokens)

                        timings = chunk_json.get("timings")
                        if timings:
                            prompt_t_s = timings.get("prompt_per_second", prompt_t_s)
                            gen_t_s = timings.get("predicted_per_second", gen_t_s)
                    except Exception:
                        pass
            
            # Send terminating chunk
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except Exception:
            pass

        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        full_content = "".join(accumulated_content)
        full_reasoning = "".join(accumulated_reasoning)
        response_preview = (full_content if full_content else full_reasoning)[:150].strip()

        if completion_tokens == 0:
            # Fallback token estimate from words if engine didn't emit final usage block in stream
            words = (len(full_content) + len(full_reasoning)) // 4
            completion_tokens = max(1, words)
            total_tokens = prompt_tokens + completion_tokens

        record = {
            "id": req_id,
            "timestamp_utc": timestamp_utc,
            "endpoint": path,
            "method": "POST",
            "status_code": 200,
            "model_alias": model_alias,
            "model_path": model_path,
            "weight_quant": weight_quant,
            "kv_quant": kv_quant,
            "n_ctx": n_ctx,
            "engine_build": engine_build,
            "vram_used_gib": vram_gib,
            "gtt_used_gib": gtt_gib,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": len(full_reasoning.split()) if full_reasoning else 0,
            "cached_tokens": 0,
            "total_tokens": total_tokens,
            "finish_reason": finish_reason or "stop",
            "duration_ms": duration_ms,
            "ttft_ms": ttft_ms or duration_ms,
            "prompt_tok_per_sec": prompt_t_s,
            "generation_tok_per_sec": gen_t_s,
            "is_stream": 1,
            "prompt_preview": prompt_preview,
            "response_preview": response_preview,
        }
        self.server.db_logger.log(record)

    def handle_passthrough(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else None
            resp = self.forward_request(body=body)
            
            self.send_response(resp.status)
            for k, v in resp.headers.items():
                if k.lower() not in ("transfer-encoding", "content-length"):
                    self.send_header(k, v)
            data = resp.read()
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() not in ("transfer-encoding", "content-length"):
                    self.send_header(k, v)
            err_data = e.read()
            self.send_header("Content-Length", str(len(err_data)))
            self.end_headers()
            self.wfile.write(err_data)
        except Exception as e:
            self.send_error(502, f"Bad Gateway: {e}")


class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    parser = argparse.ArgumentParser(description="Transparent Token Tracking Gateway")
    parser.add_argument("--port", type=int, default=8080, help="Public listening port")
    parser.add_argument("--backend-port", type=int, default=8085, help="Backend llama-server port")
    parser.add_argument("--db-path", type=Path, default=DB_DEFAULT, help="SQLite database path")
    args = parser.parse_args()

    db_logger = DBLogger(args.db_path)
    db_logger.start()

    server = ThreadedServer(("0.0.0.0", args.port), TrackerHandler)
    server.backend_port = args.backend_port
    server.db_logger = db_logger

    sys.stderr.write(f"[tracker] Listening on http://0.0.0.0:{args.port} -> backend 127.0.0.1:{args.backend_port}\n")
    sys.stderr.write(f"[tracker] Logging usage to {args.db_path}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
