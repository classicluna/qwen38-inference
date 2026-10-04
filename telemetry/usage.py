#!/usr/bin/env python3
"""CLI utility to query and inspect local token usage from SQLite."""
import argparse
import csv
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_DEFAULT = ROOT / "telemetry" / "usage.db"


def get_conn(db_path=DB_DEFAULT):
    if not Path(db_path).exists():
        sys.stderr.write(f"Database not found at {db_path}. No requests logged yet.\n")
        sys.exit(1)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def cmd_summary(args):
    conn = get_conn(args.db)
    cur = conn.cursor()

    # Overall totals
    cur.execute("""
        SELECT 
            COUNT(*) as total_reqs,
            COALESCE(SUM(prompt_tokens), 0) as total_prompt,
            COALESCE(SUM(completion_tokens), 0) as total_comp,
            COALESCE(SUM(reasoning_tokens), 0) as total_reasoning,
            COALESCE(SUM(total_tokens), 0) as grand_total,
            COALESCE(AVG(duration_ms), 0) as avg_duration,
            COALESCE(AVG(ttft_ms), 0) as avg_ttft,
            COALESCE(AVG(generation_tok_per_sec), 0) as avg_gen_speed,
            COALESCE(AVG(prompt_tok_per_sec), 0) as avg_prefill_speed
        FROM requests
        WHERE status_code = 200
    """)
    totals = cur.fetchone()

    if totals["total_reqs"] == 0:
        print("No successful requests recorded yet.")
        return

    print("=================================================================")
    print("                 LOCAL AI TOKEN USAGE SUMMARY                    ")
    print("=================================================================")
    print(f"Total Requests:       {totals['total_reqs']:,}")
    print(f"Prompt Tokens:        {totals['total_prompt']:,}")
    print(f"Completion Tokens:    {totals['total_comp']:,}")
    print(f"Reasoning Tokens:     {totals['total_reasoning']:,} (internal thinking)")
    print(f"Grand Total Tokens:   {totals['grand_total']:,}")
    print("-----------------------------------------------------------------")
    print(f"Avg TTFT:             {totals['avg_ttft']:.1f} ms")
    print(f"Avg Duration:         {totals['avg_duration']/1000:.2f} s")
    print(f"Avg Generation Speed: {totals['avg_gen_speed']:.2f} tok/s")
    print(f"Avg Prefill Speed:    {totals['avg_prefill_speed']:.2f} tok/s")
    print("=================================================================")

    # Breakdown by model & quant
    print("\n[Breakdown by Model & Quantization]")
    cur.execute("""
        SELECT 
            model_alias,
            weight_quant,
            kv_quant,
            COUNT(*) as reqs,
            SUM(prompt_tokens) as p_tokens,
            SUM(completion_tokens) as c_tokens,
            AVG(generation_tok_per_sec) as gen_speed
        FROM requests
        WHERE status_code = 200
        GROUP BY model_alias, weight_quant, kv_quant
        ORDER BY reqs DESC
    """)
    rows = cur.fetchall()
    header = f"{'Model':<16} | {'Weight Quant':<24} | {'KV':<6} | {'Reqs':<6} | {'Prompt':<10} | {'Comp':<10} | {'Tok/s':<6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        m = (r["model_alias"] or "unknown")[:16]
        wq = (r["weight_quant"] or "unknown")[:24]
        kv = (r["kv_quant"] or "")[:6]
        print(f"{m:<16} | {wq:<24} | {kv:<6} | {r['reqs']:<6} | {r['p_tokens'] or 0:<10,} | {r['c_tokens'] or 0:<10,} | {r['gen_speed'] or 0.0:<6.2f}")


def cmd_recent(args):
    conn = get_conn(args.db)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT 
            timestamp_utc,
            model_alias,
            weight_quant,
            kv_quant,
            prompt_tokens,
            completion_tokens,
            duration_ms,
            generation_tok_per_sec,
            prompt_preview,
            response_preview
        FROM requests
        ORDER BY timestamp_utc DESC
        LIMIT {int(args.limit)}
    """)
    rows = cur.fetchall()

    if not rows:
        print("No requests logged yet.")
        return

    print(f"{'Time (UTC)':<19} | {'Model':<12} | {'KV':<5} | {'Tokens (P/C)':<14} | {'Time':<7} | {'Tok/s':<6} | {'Preview'}")
    print("-" * 100)
    for r in rows:
        ts = (r["timestamp_utc"] or "")[:19].replace("T", " ")
        m = (r["model_alias"] or "")[:12]
        kv = (r["kv_quant"] or "")[:5]
        toks = f"{r['prompt_tokens']}/{r['completion_tokens']}"
        sec = f"{(r['duration_ms'] or 0)/1000:.1f}s"
        tps = f"{r['generation_tok_per_sec'] or 0:.1f}"
        prev = (r["prompt_preview"] or r["response_preview"] or "")[:35].replace("\n", " ")
        print(f"{ts:<19} | {m:<12} | {kv:<5} | {toks:<14} | {sec:<7} | {tps:<6} | {prev}")


def cmd_tail(args):
    conn = get_conn(args.db)
    print("Watching live requests (Ctrl+C to stop)...")
    last_id = None
    try:
        while True:
            cur = conn.cursor()
            if last_id is None:
                cur.execute("SELECT id FROM requests ORDER BY timestamp_utc DESC LIMIT 1")
                row = cur.fetchone()
                last_id = row["id"] if row else ""
            
            cur.execute("""
                SELECT id, timestamp_utc, model_alias, kv_quant, prompt_tokens, completion_tokens,
                       duration_ms, generation_tok_per_sec, prompt_preview, response_preview
                FROM requests
                WHERE id != ?
                ORDER BY timestamp_utc DESC LIMIT 5
            """, (last_id,))
            rows = cur.fetchall()
            if rows:
                for r in reversed(rows):
                    last_id = r["id"]
                    ts = (r["timestamp_utc"] or "")[11:19]
                    print(f"[{ts}] {r['model_alias']} ({r['kv_quant']}) | {r['prompt_tokens']} prompt + {r['completion_tokens']} comp in {r['duration_ms']/1000:.1f}s ({r['generation_tok_per_sec']:.1f} t/s)")
                    if r["prompt_preview"]:
                        print(f"  > Prompt: {r['prompt_preview'][:80]}")
                    if r["response_preview"]:
                        print(f"  < Answer: {r['response_preview'][:80]}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


def cmd_export(args):
    conn = get_conn(args.db)
    cur = conn.cursor()
    cur.execute("SELECT * FROM requests ORDER BY timestamp_utc ASC")
    rows = cur.fetchall()
    
    out_file = args.out
    if not out_file:
        out_file = "usage_export.csv"

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(rows[0].keys() if rows else [])
        for r in rows:
            writer.writerow(list(r))
    print(f"Exported {len(rows)} requests to {out_file}")


def main():
    ap = argparse.ArgumentParser(description="Query local token usage")
    ap.add_argument("--db", type=Path, default=DB_DEFAULT, help="Database path")
    sub = ap.add_subparsers(dest="cmd")

    p_sum = sub.add_parser("summary", help="Show aggregate token usage and speeds")
    p_rec = sub.add_parser("recent", help="List recent requests")
    p_rec.add_argument("-n", "--limit", type=int, default=10, help="Number of rows")

    p_tail = sub.add_parser("tail", help="Follow requests live")
    p_exp = sub.add_parser("export", help="Export to CSV")
    p_exp.add_argument("-o", "--out", type=str, default="usage_export.csv", help="Output file")

    args = ap.parse_args()
    if not args.cmd or args.cmd == "summary":
        cmd_summary(args)
    elif args.cmd == "recent":
        cmd_recent(args)
    elif args.cmd == "tail":
        cmd_tail(args)
    elif args.cmd == "export":
        cmd_export(args)


if __name__ == "__main__":
    main()
