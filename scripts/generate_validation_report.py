#!/usr/bin/env python3
"""Generate a human-readable markdown validation report from metrics JSON."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _format_pass_fail(value: float, target_value: float, operator: str) -> str:
    """Return PASS/FAIL indicator based on operator comparison."""
    if operator == "gt":
        passed = value > target_value
    elif operator == "lt":
        passed = value < target_value
    elif operator == "gte":
        passed = value >= target_value
    elif operator == "lte":
        passed = value <= target_value
    else:
        passed = False
    return "PASS" if passed else "FAIL"


def generate_report(
    metrics: dict[str, Any],
    config_path: str = "configs/rtmpose_m_finetune_baseball.py",
    checkpoint_path: str = "models/2d_pose/rtmpose_m_finetuned.pth",
) -> str:
    """Return markdown report string from metrics dict."""
    lines: list[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines.append("# 2D Pose Validation Report")
    lines.append("")
    lines.append(f"**Generated:** {now}")
    lines.append("")

    # Traceability
    lines.append("## Traceability")
    lines.append("")
    lines.append(f"- **Config snapshot:** `{config_path}`")
    lines.append(f"- **Checkpoint:** `{checkpoint_path}`")
    lines.append("")

    # Aggregate metrics
    agg = metrics.get("aggregate", {})
    tc = agg.get("temporal_coherence", 0.0)
    ks = agg.get("keypoint_stability", 0.0)
    mr = agg.get("missing_rate", 0.0)

    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Metric | Value | Target | Status |")
    lines.append("|--------|-------|--------|--------|")
    lines.append(
        f"| Temporal Coherence | {tc:.6f} | > 0.82 | "
        f"{_format_pass_fail(tc, 0.82, 'gt')} |"
    )
    lines.append(
        f"| Keypoint Stability | {ks:.6f} | > 0.92 | "
        f"{_format_pass_fail(ks, 0.92, 'gt')} |"
    )
    lines.append(
        f"| Missing Rate | {mr:.6f} | < 0.05 | "
        f"{_format_pass_fail(mr, 0.05, 'lt')} |"
    )
    lines.append("")

    # Per-video metrics
    videos = metrics.get("videos", [])
    lines.append("## Per-Video Metrics")
    lines.append("")
    if videos:
        lines.append("| Video | Frames | TC | KS | Missing Rate |")
        lines.append("|-------|--------|-------|-------|--------------|")
        for v in videos:
            vid_tc = v.get("temporal_coherence", 0.0)
            vid_ks = v.get("keypoint_stability", 0.0)
            vid_mr = v.get("missing_rate", 0.0)
            frames = v.get("frames", 0)
            lines.append(
                f"| {v.get('video_id', 'unknown')} | {frames} | "
                f"{vid_tc:.6f} | {vid_ks:.6f} | {vid_mr:.6f} |"
            )
    else:
        lines.append("No per-video metrics available.")
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate markdown validation report from metrics JSON"
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("logs/validation_metrics.json"),
        help="Path to validation metrics JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/2d_validation_report.md"),
        help="Output markdown report path",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/rtmpose_m_finetune_baseball.py",
        help="Path to training config snapshot for traceability",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="models/2d_pose/rtmpose_m_finetuned.pth",
        help="Path to fine-tuned checkpoint for traceability",
    )
    args = parser.parse_args(argv)

    if not args.metrics_json.is_file():
        print(f"Metrics JSON not found: {args.metrics_json}", file=sys.stderr)
        return 1

    try:
        with open(args.metrics_json, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in {args.metrics_json}: {exc}", file=sys.stderr)
        return 1

    report = generate_report(
        metrics,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
