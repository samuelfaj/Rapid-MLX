#!/usr/bin/env python3
"""Run reproducible Rapid-MLX agentic benchmarks and write BENCHMARK.md."""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = Path(os.environ.get("BENCH_ARTIFACT_DIR", "/tmp/rapid-mlx-bench"))
PI_AGENT_DIR = ARTIFACT_ROOT / "pi-agent"
PORT = int(os.environ.get("BENCH_PORT", "8010"))

DEFAULT_PROMPT = os.environ.get("BENCH_PROMPT")

MODEL_35B = "/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit"
DRAFTER_35B = "/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-DFlash"
MODEL_27B = "/Users/samuelfajreldines/dev/models/Qwen3.6-27B-UD-Q4_K_XL-mlx"
DRAFTER_27B = "/Users/samuelfajreldines/dev/models/Qwen3.6-27B-DFlash"
SPEC_PREFILL_DRAFT = "/Users/samuelfajreldines/dev/models/Qwen3-1.7B-4bit-mlx"


@dataclass(frozen=True)
class Profile:
    key: str
    model_label: str
    mode: str
    target: str
    drafter: str | None
    optimized: bool


PROFILES = [
    Profile("qwen36_35b_baseline", "Qwen3.6 35B A3B 4bit", "baseline", MODEL_35B, None, False),
    Profile("qwen36_35b_optimized", "Qwen3.6 35B A3B 4bit", "optimized", MODEL_35B, DRAFTER_35B, True),
    Profile("qwen36_27b_baseline", "Qwen3.6 27B UD Q4_K_XL", "baseline", MODEL_27B, None, False),
    Profile("qwen36_27b_optimized", "Qwen3.6 27B UD Q4_K_XL", "optimized", MODEL_27B, DRAFTER_27B, True),
]


def write_pi_config() -> None:
    PI_AGENT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "providers": {
            "rapid-mlx": {
                "api": "openai-completions",
                "apiKey": "local",
                "baseUrl": f"http://127.0.0.1:{PORT}/v1",
                "compat": {
                    "supportsStore": False,
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                    "supportsStrictMode": False,
                    "maxTokensField": "max_tokens",
                    "thinkingFormat": "qwen-chat-template",
                },
                "models": [
                    {
                        "id": "local",
                        "name": "Rapid-MLX local",
                        "api": "openai-completions",
                        "reasoning": False,
                        "input": ["text"],
                        "contextWindow": 65536,
                        "maxTokens": 4096,
                    }
                ],
            }
        }
    }
    (PI_AGENT_DIR / "models.json").write_text(json.dumps(config, indent=2) + "\n")


def http_json(path: str, timeout: float = 5.0) -> Any:
    url = f"http://127.0.0.1:{PORT}{path}"
    req = urllib.request.Request(url, headers={"accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    data = path.read_text(errors="replace").splitlines()
    return "\n".join(data[-lines:])


def server_command(profile: Profile) -> list[str]:
    cmd = ["uv", "run", "rapid-mlx", "serve", profile.target]
    if profile.optimized:
        if not profile.drafter:
            raise ValueError(f"{profile.key} needs drafter")
        cmd.extend(
            [
                "--drafter",
                profile.drafter,
                "--dflash-ddtree-budget",
                "4",
                "--dflash-no-adaptive",
                "--dflash-fallback-mode",
                "ngram",
                "--thinking-ngram",
                "--ngram-num-draft-tokens",
                "4",
                "--ngram-size",
                "2",
                "--ngram-min-matches",
                "1",
                "--structured-cot-tools",
                "--agentic-guard",
                "--pin-system-prompt",
                "--speculative-prefill",
                "--speculative-prefill-draft-model",
                SPEC_PREFILL_DRAFT,
                "--speculative-prefill-ratio",
                "0.85",
            ]
        )
    cmd.extend(
        [
            "--served-model-name",
            "local",
            "--port",
            str(PORT),
            "--default-temperature",
            "0",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder_xml",
            "--max-tokens",
            "4096",
            "--timeout",
            "300",
        ]
    )
    return cmd


def server_env(profile: Profile) -> dict[str, str]:
    env = os.environ.copy()
    if profile.optimized:
        env["DFLASH_DRAFT_SINK"] = "64"
        env["DFLASH_DRAFT_WINDOW"] = "1024"
    return env


def start_server(profile: Profile, profile_dir: Path) -> tuple[subprocess.Popen[bytes], Path]:
    log_path = profile_dir / "server.log"
    log_file = log_path.open("wb")
    proc = subprocess.Popen(
        server_command(profile),
        cwd=ROOT,
        env=server_env(profile),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._benchmark_log_file = log_file  # type: ignore[attr-defined]
    return proc, log_path


def stop_server(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None:
        return
    log_file = getattr(proc, "_benchmark_log_file", None)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(timeout=45)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=15)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=5)
    if log_file:
        log_file.close()


def wait_for_health(proc: subprocess.Popen[bytes], log_path: Path, timeout: float = 300.0) -> Any:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}\n{tail(log_path)}")
        try:
            health = http_json("/health", timeout=3)
            if health.get("status") == "healthy" and health.get("model_loaded"):
                return health
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"server health timeout: {last_error}\n{tail(log_path)}")


def request_entries() -> list[dict[str, Any]]:
    try:
        data = http_json("/v1/requests", timeout=5)
    except Exception:
        return []
    if isinstance(data, dict) and isinstance(data.get("entries"), list):
        return [entry for entry in data["entries"] if isinstance(entry, dict)]
    return []


def diff_entries(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(after) >= len(before):
        return after[len(before) :]
    before_ids = {json.dumps(entry, sort_keys=True, default=str) for entry in before}
    return [entry for entry in after if json.dumps(entry, sort_keys=True, default=str) not in before_ids]


def merge_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        key = str(entry.get("request_id") or json.dumps(entry, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        merged.append(entry)
    return merged


def find_numbers(obj: Any, wanted: set[str]) -> list[float]:
    found: list[float] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            normalized = key.lower().replace("-", "_")
            if normalized in wanted and isinstance(value, (int, float)):
                found.append(float(value))
            found.extend(find_numbers(value, wanted))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(find_numbers(item, wanted))
    return found


def summarize_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    output_keys = {"output_tokens", "completion_tokens", "generated_tokens", "tokens_generated"}
    input_keys = {"input_tokens", "prompt_tokens", "prefill_tokens"}
    tps_keys = {"tokens_per_second", "tok_s", "tokens_s", "decode_tps"}
    ttft_keys = {"ttft", "ttft_seconds", "time_to_first_token", "first_token_seconds"}
    output_tokens = find_numbers(entries, output_keys)
    input_tokens = find_numbers(entries, input_keys)
    tps = find_numbers(entries, tps_keys)
    ttft = find_numbers(entries, ttft_keys)
    return {
        "request_count": len(entries),
        "input_tokens_sum": int(sum(input_tokens)) if input_tokens else None,
        "output_tokens_sum": int(sum(output_tokens)) if output_tokens else None,
        "tokens_per_second_median": round(statistics.median(tps), 2) if tps else None,
        "ttft_seconds_median": round(statistics.median(ttft), 3) if ttft else None,
        "raw_entries_sample": entries[-3:],
    }


def run_logged(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    timeout: int,
    env: dict[str, str] | None = None,
    request_poll_interval: float | None = None,
) -> dict[str, Any]:
    start = time.monotonic()
    timed_out = False
    polled_entries: list[dict[str, Any]] = []
    with log_path.open("wb") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = start + timeout
        next_poll = start
        while proc.poll() is None:
            now = time.monotonic()
            if request_poll_interval and now >= next_poll:
                polled_entries.extend(request_entries())
                next_poll = now + request_poll_interval
            if now >= deadline:
                timed_out = True
                break
            time.sleep(0.5)
        if timed_out:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=15)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
    return {
        "command": cmd,
        "cwd": str(cwd),
        "log_path": str(log_path),
        "exit_code": proc.returncode,
        "timeout": timed_out,
        "wall_seconds": round(time.monotonic() - start, 2),
        "polled_request_count": len(merge_entries(polled_entries)),
        "polled_request_entries": merge_entries(polled_entries),
    }


def detect_project_root(work_dir: Path) -> Path | None:
    packages = [path for path in work_dir.rglob("package.json") if "node_modules" not in path.parts]
    if not packages:
        return None
    packages.sort(key=lambda p: (len(p.relative_to(work_dir).parts), str(p)))
    return packages[0].parent


def count_files(path: Path, patterns: tuple[str, ...]) -> int:
    total = 0
    for pattern in patterns:
        total += sum(1 for _ in path.rglob(pattern))
    return total


def package_manager(project_root: Path, package: dict[str, Any]) -> str:
    declared = str(package.get("packageManager") or "").lower()
    if declared.startswith("bun@") or (project_root / "bun.lockb").exists() or (project_root / "bun.lock").exists():
        return "bun"
    if declared.startswith("pnpm@") or (project_root / "pnpm-lock.yaml").exists():
        return "pnpm"
    if declared.startswith("yarn@") or (project_root / "yarn.lock").exists():
        return "yarn"
    return "npm"


def install_command(manager: str) -> list[str]:
    if manager == "bun":
        return ["bun", "install"]
    if manager == "pnpm":
        return ["pnpm", "install"]
    if manager == "yarn":
        return ["yarn", "install"]
    return ["npm", "install", "--no-audit", "--no-fund"]


def run_script_command(manager: str, script: str, extra: list[str] | None = None) -> list[str]:
    extra = extra or []
    if manager == "bun":
        base = ["bun", "run", script]
    elif manager == "pnpm":
        base = ["pnpm", "run", script]
    elif manager == "yarn":
        base = ["yarn", script]
    else:
        base = ["npm", "run", script]
    return base + extra


def validation_requirements(prompt: str, scripts: dict[str, Any]) -> set[str]:
    lowered = prompt.lower()
    required = {"install"}
    if any(term in lowered for term in ("test", "tests", "tdd", "unit")):
        required.add("test")
    if "build" in lowered and "build" in scripts:
        required.add("build")
    return required


def validate_project(project_root: Path | None, run_dir: Path, timeout: int, prompt: str) -> dict[str, Any]:
    if project_root is None:
        return {"success": False, "reason": "no package.json found", "commands": [], "project": None}
    package = json.loads((project_root / "package.json").read_text())
    scripts = package.get("scripts", {})
    manager = package_manager(project_root, package)
    commands: list[tuple[str, list[str]]] = [("install", install_command(manager))]
    if "test" in scripts:
        test_cmd = run_script_command(manager, "test")
        if "vitest" in str(scripts["test"]).lower():
            test_cmd = run_script_command(manager, "test", ["--", "--run"])
        commands.append(("test", test_cmd))
    if "build" in scripts:
        commands.append(("build", run_script_command(manager, "build")))
    if "lint" in scripts:
        commands.append(("lint", run_script_command(manager, "lint")))

    results: list[dict[str, Any]] = []
    for name, cmd in commands:
        results.append(run_logged(cmd, project_root, run_dir / f"validation-{name}.log", timeout))

    required = validation_requirements(prompt, scripts)
    passed = {name for name, _ in commands if any(r["command"] == _ and r["exit_code"] == 0 and not r["timeout"] for r in results)}
    success = required.issubset(passed) and all(not r["timeout"] and r["exit_code"] == 0 for r in results)
    return {
        "success": success,
        "reason": None if success else "validation command failed or required script missing",
        "commands": results,
        "project": {
            "root": str(project_root),
            "package_name": package.get("name"),
            "package_manager": manager,
            "scripts": scripts,
            "file_count": count_files(project_root, ("*.ts", "*.tsx", "*.css", "*.json")),
            "test_file_count": count_files(project_root, ("*.test.ts", "*.test.tsx", "*.spec.ts", "*.spec.tsx")),
        },
    }


def run_pi_once(profile: Profile, run_number: int, args: argparse.Namespace) -> dict[str, Any]:
    run_dir = ARTIFACT_ROOT / profile.key / f"run-{run_number}"
    work_dir = run_dir / "work"
    if run_dir.exists():
        subprocess.run(["rm", "-rf", str(run_dir)], check=True)
    work_dir.mkdir(parents=True)

    env = os.environ.copy()
    env["PI_CODING_AGENT_DIR"] = str(PI_AGENT_DIR)
    env["PI_OFFLINE"] = "1"
    env.setdefault("NO_COLOR", "1")

    before = request_entries()
    pi_result = run_logged(
        [
            "pi",
            "--provider",
            "rapid-mlx",
            "--model",
            "local",
            "--no-session",
            "--no-context-files",
            "--api-key",
            "local",
            "-p",
            args.prompt,
        ],
        work_dir,
        run_dir / "pi.log",
        args.pi_timeout,
        env,
        request_poll_interval=args.metrics_poll_interval,
    )
    after = request_entries()
    entries = merge_entries(diff_entries(before, after) + pi_result.pop("polled_request_entries", []))
    project_root = detect_project_root(work_dir)
    validation = validate_project(project_root, run_dir, args.validation_timeout, args.prompt)
    return {
        "run_number": run_number,
        "prompt": args.prompt,
        "work_dir": str(work_dir),
        "pi": pi_result,
        "request_metrics": summarize_entries(entries),
        "validation": validation,
    }


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.median(values), 2)


def profile_summary(profile: dict[str, Any]) -> dict[str, Any]:
    runs = profile.get("runs", [])
    pi_done = [r for r in runs if not r["pi"]["timeout"] and r["pi"]["exit_code"] == 0]
    valid = [r for r in pi_done if r["validation"]["success"]]
    walls = [float(r["pi"]["wall_seconds"]) for r in runs]
    valid_walls = [float(r["pi"]["wall_seconds"]) for r in valid]
    tps = [
        float(r["request_metrics"]["tokens_per_second_median"])
        for r in runs
        if r["request_metrics"].get("tokens_per_second_median") is not None
        and (r["request_metrics"].get("output_tokens_sum") or 0) > 0
    ]
    return {
        "runs": len(runs),
        "pi_finished": len(pi_done),
        "validation_success": len(valid),
        "timeouts": sum(1 for r in runs if r["pi"]["timeout"]),
        "median_wall_seconds": median(walls),
        "median_valid_wall_seconds": median(valid_walls),
        "median_tokens_per_second": median(tps),
    }


def speedup_line(results: list[dict[str, Any]], model_label: str) -> str:
    base = next((p for p in results if p["profile"]["model_label"] == model_label and p["profile"]["mode"] == "baseline"), None)
    opt = next((p for p in results if p["profile"]["model_label"] == model_label and p["profile"]["mode"] == "optimized"), None)
    if not base or not opt:
        return f"- {model_label}: sem comparacao completa."
    bsum = profile_summary(base)
    osum = profile_summary(opt)
    decode_ratio = None
    if bsum["median_tokens_per_second"] and osum["median_tokens_per_second"]:
        decode_ratio = round(float(osum["median_tokens_per_second"]) / float(bsum["median_tokens_per_second"]), 2)
    decode_text = f"; throughput de decode {decode_ratio}x maior" if decode_ratio else ""
    if bsum["validation_success"] == 0 and osum["validation_success"] == 0:
        return f"- {model_label}: speedup fim-a-fim = n/a; 0/{bsum['runs']} baseline e 0/{osum['runs']} otimizado validaram{decode_text}."
    b_ref = bsum["median_valid_wall_seconds"] or bsum["median_wall_seconds"]
    o_ref = osum["median_valid_wall_seconds"] or osum["median_wall_seconds"]
    if not b_ref or not o_ref:
        return f"- {model_label}: sem mediana suficiente."
    ratio = round(float(b_ref) / float(o_ref), 2)
    lower = ">= " if bsum["validation_success"] == 0 and osum["validation_success"] > 0 else ""
    return f"- {model_label}: otimizado {lower}{ratio}x mais rapido por mediana wall-clock."


def format_seconds(value: Any) -> str:
    if value is None:
        return "n/a"
    seconds = float(value)
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    if minutes:
        return f"{minutes}m{rest:04.1f}s"
    return f"{rest:.1f}s"


def write_markdown(results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    summaries = [(item, profile_summary(item)) for item in results]
    total_runs = sum(summary["runs"] for _, summary in summaries)
    total_valid = sum(summary["validation_success"] for _, summary in summaries)
    optimized_valid = sum(
        summary["validation_success"]
        for item, summary in summaries
        if item["profile"]["optimized"]
    )
    baseline_valid = sum(
        summary["validation_success"]
        for item, summary in summaries
        if not item["profile"]["optimized"]
    )
    if total_runs and total_valid == total_runs:
        conclusion = (
            "5/ Conclusao: todos os perfis validaram. Compare a linha de speedup "
            "por modelo para confirmar aceleracao fim-a-fim com qualidade igual."
        )
    elif optimized_valid and optimized_valid == baseline_valid:
        conclusion = (
            "5/ Conclusao: baseline e otimizado tiveram a mesma quantidade de "
            "runs validados, mas nem todos os runs passaram; qualidade ainda exige "
            "analise dos artefatos brutos."
        )
    elif optimized_valid:
        conclusion = (
            "5/ Conclusao: o perfil otimizado validou mais runs que o baseline "
            "neste conjunto, mas qualidade igual so deve ser aceita quando os dois "
            "perfis validarem os mesmos requisitos."
        )
    else:
        conclusion = (
            "5/ Conclusao: nenhum perfil otimizado atingiu sucesso completo. "
            "Neste teste, throughput parcial nao basta se agente nao termina e "
            "valida."
        )
    lines = [
        "# Benchmark Rapid-MLX: agente local sem vs com otimizacao",
        "",
        "1/ Mesmo prompt, mesma maquina, mesma porta, pasta vazia por run.",
        "",
        f"> Prompt: `{args.prompt}`",
        "",
        "2/ O teste nao mede so tokens/s. Mede comportamento agentico: criar o artefato pedido em pasta limpa, terminar sozinho, depois passar em validacao local.",
        "",
        "## Resultado curto",
        "",
    ]
    for model_label in sorted({item["profile"]["model_label"] for item in results}):
        lines.append(speedup_line(results, model_label))
    lines.extend(
        [
            "",
            "## Tabela",
            "",
            "| modelo | perfil | runs | pi finalizou | validou | timeouts | mediana wall | mediana wall valida | tok/s mediana |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item, summary in summaries:
        profile = item["profile"]
        lines.append(
            "| {model} | {mode} | {runs} | {finished}/{runs} | {valid}/{runs} | {timeouts} | {wall} | {valid_wall} | {tps} |".format(
                model=profile["model_label"],
                mode=profile["mode"],
                runs=summary["runs"],
                finished=summary["pi_finished"],
                valid=summary["validation_success"],
                timeouts=summary["timeouts"],
                wall=format_seconds(summary["median_wall_seconds"]),
                valid_wall=format_seconds(summary["median_valid_wall_seconds"]),
                tps=summary["median_tokens_per_second"] if summary["median_tokens_per_second"] is not None else "n/a",
            )
        )
    lines.extend(
        [
            "",
            "## Metodo",
            "",
            f"- Runs por perfil: `{args.runs}`.",
            f"- Timeout do agente: `{args.pi_timeout}s`.",
            f"- Timeout por comando de validacao: `{args.validation_timeout}s`.",
            "- Baseline: target model sem drafter, sem DDTree, sem ngram fallback, sem structured-cot tool guard.",
            "- Otimizado: target model + drafter DFlash pareado, Speculative Prefill conservador com draft pequeno, DDTree budget 4, adaptive off, fallback ngram, thinking ngram, structured-cot e structured-cot-tools.",
            "- `pi` usa provider local OpenAI-compatible via `PI_CODING_AGENT_DIR`, `rapid-mlx` em `http://127.0.0.1:8010/v1`, `temperature=0`, `max_tokens=4096`.",
            "- Validacao: instala dependencias com package manager detectado, roda `test` quando existir ou for pedido, roda `build`/`lint` quando existirem.",
            "- Tok/s e diagnostico de servidor, nao criterio de sucesso. Runs longos podem trocar entradas antigas do `/v1/requests`; `BENCHMARK.py` agora faz polling para novos reruns.",
            "",
            "## Perfis",
            "",
        ]
    )
    for item in results:
        profile = item["profile"]
        cmd = " ".join(item.get("server_command", []))
        lines.extend(
            [
                f"### {profile['key']}",
                "",
                f"- Target: `{profile['target']}`.",
                f"- Drafter: `{profile['drafter'] or 'none'}`.",
                f"- Server command: `{cmd}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Leitura para X",
            "",
            "3/ Otimizacao que importa aqui nao e micro-benchmark isolado. E fim-a-fim: se agente para cedo, entra em loop, ou gera projeto quebrado, velocidade de decode nao salva run.",
            "",
            "4/ Resultado bom = run termina, projeto nasce em pasta limpa, testes passam, build passa. Resultado ruim = timeout, erro de tool-call, pacote incompleto, ou validacao quebrada.",
            "",
            conclusion,
            "",
            f"6/ Artefatos brutos: `{ARTIFACT_ROOT}`. JSON completo: `{ARTIFACT_ROOT / 'results.json'}`.",
            "",
        ]
    )
    (ROOT / "BENCHMARK.md").write_text("\n".join(lines))


def run_profile(profile: Profile, args: argparse.Namespace) -> dict[str, Any]:
    paths = [Path(profile.target), *([Path(profile.drafter)] if profile.drafter else [])]
    if profile.optimized:
        paths.append(Path(SPEC_PREFILL_DRAFT))
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"missing model path for {profile.key}: {path}")

    profile_dir = ARTIFACT_ROOT / profile.key
    profile_dir.mkdir(parents=True, exist_ok=True)
    server_proc: subprocess.Popen[bytes] | None = None
    server_log = profile_dir / "server.log"
    result: dict[str, Any] = {
        "profile": asdict(profile),
        "server_command": server_command(profile),
        "server_log": str(server_log),
        "server_health": None,
        "runs": [],
        "error": None,
    }
    try:
        print(f"[server] start {profile.key}", flush=True)
        server_proc, server_log = start_server(profile, profile_dir)
        result["server_log"] = str(server_log)
        result["server_health"] = wait_for_health(server_proc, server_log)
        print(f"[server] ready {profile.key}", flush=True)
        for index in range(1, args.runs + 1):
            print(f"[run] {profile.key} #{index}", flush=True)
            result["runs"].append(run_pi_once(profile, index, args))
            summary = profile_summary(result)
            print(
                f"[run] {profile.key} #{index} done "
                f"finished={summary['pi_finished']}/{summary['runs']} "
                f"valid={summary['validation_success']}/{summary['runs']} "
                f"median={format_seconds(summary['median_wall_seconds'])}",
                flush=True,
            )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[error] {profile.key}: {result['error']}", flush=True)
    finally:
        stop_server(server_proc)
        print(f"[server] stop {profile.key}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--runs", type=int, default=int(os.environ.get("BENCH_RUNS", "3")))
    parser.add_argument("--pi-timeout", type=int, default=int(os.environ.get("BENCH_PI_TIMEOUT", "600")))
    parser.add_argument("--validation-timeout", type=int, default=int(os.environ.get("BENCH_VALIDATION_TIMEOUT", "180")))
    parser.add_argument(
        "--metrics-poll-interval",
        type=float,
        default=float(os.environ.get("BENCH_METRICS_POLL_INTERVAL", "10")),
    )
    parser.add_argument("--render-only", action="store_true", help="Render BENCHMARK.md from existing results.json")
    parser.add_argument(
        "--profiles",
        nargs="*",
        default=[profile.key for profile in PROFILES],
        choices=[profile.key for profile in PROFILES],
    )
    args = parser.parse_args()
    if not args.prompt:
        parser.error("provide --prompt or BENCH_PROMPT")

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    write_pi_config()
    if args.render_only:
        results_path = ARTIFACT_ROOT / "results.json"
        results = json.loads(results_path.read_text())
        write_markdown(results, args)
        print(f"[done] rendered {ROOT / 'BENCHMARK.md'} from {results_path}", flush=True)
        return 0
    selected = [profile for profile in PROFILES if profile.key in set(args.profiles)]
    results = [run_profile(profile, args) for profile in selected]
    (ARTIFACT_ROOT / "results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    write_markdown(results, args)
    print(f"[done] wrote {ROOT / 'BENCHMARK.md'}", flush=True)
    print(f"[done] wrote {ARTIFACT_ROOT / 'results.json'}", flush=True)
    return 0 if all(not item.get("error") for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
