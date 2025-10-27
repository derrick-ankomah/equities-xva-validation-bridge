import re, json, os, sys, pathlib

root = pathlib.Path(__file__).resolve().parent.parent
reports = root / "reports"

def parse_robustness():
    log = (reports/"robustness"/"run.log").read_text(errors="ignore")
    local = re.search(r"Local drift@eps=.*?:\s*mean=([0-9.\-eE]+),\s*p95=([0-9.\-eE]+)", log)
    glob  = re.search(r"Global drift:\s*mean=([0-9.\-eE]+),\s*p95=([0-9.\-eE]+)", log)
    verdict = "PASS" if "PASS" in log.upper() and "ROBUSTNESS PASS" in log.upper() else ("FAIL" if "FAIL" in log.upper() else "UNKNOWN")
    d = {}
    if local: d.update({"local_mean": float(local.group(1)), "local_p95": float(local.group(2))})
    if glob:  d.update({"global_mean": float(glob.group(1)), "global_p95": float(glob.group(2))})
    d["result"] = verdict

    (reports/"robustness"/"summary.json").write_text(json.dumps(d, indent=2))
    md = ["# Robustness Summary", ""]
    if "local_mean" in d:
        md.append(f"- Local drift: mean = {d['local_mean']:.6f}, p95 = {d['local_p95']:.6f}")
    if "global_mean" in d:
        md.append(f"- Global drift: mean = {d['global_mean']:.6f}, p95 = {d['global_p95']:.6f}")
    md.append(f"- Result: **{verdict}**")
    (reports/"robustness"/"README.md").write_text("\n".join(md)+"\n")

def parse_uncertainty():
    log = (reports/"uncertainty"/"run.log").read_text(errors="ignore")
    q = re.search(r"Conformal radius q=([0-9.\-eE]+)", log)
    d = {}
    if q: d["conformal_radius"] = float(q.group(1))
    (reports/"uncertainty"/"summary.json").write_text(json.dumps(d, indent=2))
    md = ["# Uncertainty Summary", ""]
    if q:
        md.append(f"- Conformal radius **q = {d['conformal_radius']:.6f}**")
    else:
        md.append("- Conformal radius not found in log.")
    (reports/"uncertainty"/"README.md").write_text("\n".join(md)+"\n")

def parse_offset():
    log_path = reports/"offset"/"run.log"
    if not log_path.exists():
        return
    log = log_path.read_text(errors="ignore")

    # Try a few patterns for R^2 messages
    base  = re.search(r"Base\s*R2\s*=?\s*([0-9.\-eE]+)", log, re.IGNORECASE)
    fixed = re.search(r"(Fixed|Offset(ed)?)\s*R2\s*=?\s*([0-9.\-eE]+)", log, re.IGNORECASE)

    d = {}
    if base:  d["base_r2"]  = float(base.group(1))
    if fixed: d["fixed_r2"] = float(fixed.group(3))

    (reports/"offset"/"summary.json").write_text(json.dumps(d, indent=2))

    md = ["# Offset / Residual Summary", ""]
    if "base_r2" in d or "fixed_r2" in d:
        md.append(f"- Base R² = {d.get('base_r2','?')}")
        md.append(f"- Fixed R² = {d.get('fixed_r2','?')}")
    else:
        md.append("- R² metrics not found in log (script may have printed different text).")
    (reports/"offset"/"README.md").write_text("\n".join(md)+"\n")

def main():
    parse_robustness()
    parse_uncertainty()
    parse_offset()
    print("Parsed logs into reports/*/README.md and summary.json")

if __name__ == "__main__":
    main()
