"""Bounded local run with retained command, source hashes, output and termination state."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        parser.error("run-id must contain only letters, digits, hyphens or underscores")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.timeout <= 0:
        parser.error("Provide a Python command and positive timeout")
    root = Path(__file__).resolve().parents[1]
    output = root / "runs" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", *command]
    (output / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    files = [*root.glob("imdb2011/*.py"), *root.glob("configs/*.json"),
             *root.glob("tests/*.py"), root / "requirements-lock.txt", Path(__file__)]
    (output / "source_hashes.json").write_text(json.dumps(
        {str(p.relative_to(root)): digest(p) for p in files}, indent=2), encoding="utf-8")
    started = None
    ended = None

    def record(status, reason, execution_claim):
        def q(value):
            return json.dumps(value, ensure_ascii=False)
        text = f'''schema_version: "1.0"
contract_version: "2.0"
run_id: {q(args.run_id)}
parent_run_id: null
experiment_plan_id: imdb2011-code-validation
plan_version: "1"
experiment_mode: COMPUTATIONAL_AI
run_status: {status}
execution_claim: {str(execution_claim).lower()}
started_at: {q(started)}
ended_at: {q(ended)}
responsible_role: local-assistant
inputs:
  input_locators:
    - ../../configs
  data_or_material_versions:
    - aclImdb_v1
  source_or_protocol_locators:
    - ../../imdb2011
    - source_hashes.json
  configuration_locators:
    - command.json
execution:
  planned_command_or_procedure: See command.json
  actual_command_or_procedure_locator: command.json
  environment_or_instrument_locator: ../../requirements-lock.txt
  software_or_instrument_versions:
    - ../../requirements-lock.txt
  seeds:
    - 42
  nondeterminism_notes:
    - Platform and BLAS differences may change floating-point results
hardware_evidence:
  discovery_source: USER_REPORTED
  evidence_locator: user:conversation-hardware
  read_only_command: null
  command_output_locators: []
  confirmation_status: USER_CONFIRMED
  confirmation_locator: user:RTX3070Ti-8GB
evidence:
  log_locators:
    - stdout.log
  output_locators:
    - stdout.log
  checkpoint_locators: []
  checksums_or_stable_ids:
    - source_hashes.json
  warnings: []
  primary_observation_available: false
simulation:
  usage: {"TEST_ONLY" if args.test_only else "NONE"}
  label: {"TEST_ONLY_SIMULATION" if args.test_only else "null"}
  purpose: Code validation or diagnostic run only
  result_claim_allowed: false
deviations: []
termination_reason: {q(reason)}
invalidated_reason: null
operator_or_user_confirmation: user:continue-unfinished-implementation
integrity_ledger_entry_ids: []
'''
        (output / "run-record.yaml").write_text(text, encoding="utf-8")

    record("NOT_STARTED", None, False)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    started = stamp()
    begun = time.perf_counter()
    with (output / "stdout.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        record("RUNNING", None, True)
        try:
            code = process.wait(timeout=args.timeout)
            status, reason = ("SUCCEEDED", "exit 0") if code == 0 else ("FAILED", f"exit {code}")
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            process.terminate()
            process.wait()
            code, status, reason = 124, "INTERRUPTED", type(error).__name__
        ended = stamp()
        record(status, reason, True)
    print((output / "stdout.log").read_text(encoding="utf-8", errors="replace"))
    print(json.dumps({"status": status, "seconds": time.perf_counter() - begun,
                      "record": str(output / "run-record.yaml")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
