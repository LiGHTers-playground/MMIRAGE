"""Shared SGLang Diffusion server lifecycle for MMIRAGE orchestration."""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from contextlib import contextmanager
from typing import Any, Deque, Iterator, List, Optional

from mmirage.core.process.processors.image_gen.config import SGLangBackendConfig

logger = logging.getLogger(__name__)

MMIRAGE_SGLANG_BASE_URL = "MMIRAGE_SGLANG_BASE_URL"
DEFAULT_SGLANG_PORT = 30010
SGLANG_PORT_SEARCH_ATTEMPTS = 100
SGLANG_READINESS_POLL_SECONDS = 2.0


def get_sglang_server_config(cfg: Any) -> Optional[SGLangBackendConfig]:
    """Return the shared server config when this run uses backend='sglang'."""
    configs = [
        processor.sglang
        for processor in cfg.processors
        if getattr(processor, "type", None) == "image_gen"
        and getattr(processor, "backend", None) == "sglang"
    ]
    if not configs:
        return None
    elif len(configs) > 1:
        raise ValueError(
            "Only one backend='sglang' image_gen processor is supported per run"
        )
    else:
        return configs[0]


def launch_sglang_server(config: SGLangBackendConfig) -> subprocess.Popen[bytes]:
    """Launch the shared SGLang Diffusion server."""
    if config.port is None:
        raise RuntimeError("SGLang server port must be resolved before launch.")
    executable = shutil.which("sglang")
    if executable is None:
        raise RuntimeError(
            "Could not find the `sglang` executable. Activate or install an "
            "environment containing SGLang before running MMIRAGE."
        )
    _require_sglang_diffusion_installation()

    cmd = [
        executable,
        "serve",
        "--model-path",
        config.model_path,
        "--port",
        str(config.port),
        "--num-gpus",
        str(config.num_gpus),
    ]
    if config.dtype:
        cmd += ["--dtype", config.dtype]
    cmd += list(config.extra_server_args)

    logger.info("Starting shared SGLang Diffusion server: %s", _shell_join(cmd))
    proc: subprocess.Popen[bytes] = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    output_tail: Deque[str] = deque(maxlen=40)
    setattr(proc, "_mmirage_output_tail", output_tail)
    threading.Thread(
        target=_log_process_output,
        args=(proc, output_tail),
        daemon=True,
    ).start()
    logger.info("Shared SGLang Diffusion server started with pid=%d", proc.pid)
    return proc


def _require_sglang_diffusion_installation() -> None:
    """Raise an actionable error when the active SGLang install lacks diffusion."""
    command = [
        sys.executable,
        "-c",
        "import sglang.multimodal_gen.runtime.entrypoints.cli.serve",
    ]
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "The active SGLang 0.5.10 installation cannot import its diffusion "
            "server. Rebuild the EDF environment with a consistent "
            "`sglang[diffusion]==0.5.10` installation and verify it with "
            '`python -c "import sglang.multimodal_gen.runtime.entrypoints.cli.serve"`. '
            f"Import check output:\n{result.stderr.strip()}"
        )


def wait_for_sglang_server(
    proc: subprocess.Popen[bytes],
    config: SGLangBackendConfig,
) -> None:
    """Wait until the shared SGLang server reports readiness."""
    if config.port is None:
        raise RuntimeError(
            "SGLang server port must be resolved before readiness checks."
        )
    server_root = f"http://127.0.0.1:{config.port}"
    candidate_urls = (
        f"{server_root}/models",
        f"{server_root}/health",
        f"{server_root}/v1/models",
    )
    deadline = time.monotonic() + config.startup_timeout_seconds
    last_error = "server did not respond"

    logger.info(
        "Waiting up to %ds for shared SGLang server readiness",
        config.startup_timeout_seconds,
    )
    while time.monotonic() < deadline:
        retcode = proc.poll()
        if retcode is not None:
            raise RuntimeError(
                f"SGLang server exited before becoming ready with code {retcode}. "
                f"Recent SGLang output:\n{_format_output_tail(proc)}"
            )

        for url in candidate_urls:
            try:
                _read_json(url, config.api_key)
                logger.info("Shared SGLang server is ready at %s", url)
                return
            except Exception as exc:
                last_error = f"{url}: {exc}"
        time.sleep(SGLANG_READINESS_POLL_SECONDS)

    raise RuntimeError(
        "SGLang server did not become ready within "
        f"{config.startup_timeout_seconds}s. Last readiness error: {last_error}\n"
        f"Recent SGLang output:\n{_format_output_tail(proc)}"
    )


def stop_sglang_server(proc: subprocess.Popen[bytes], grace_seconds: int = 30) -> None:
    """Stop the shared SGLang server process group."""
    if proc.poll() is not None:
        return

    logger.info("Stopping shared SGLang server with pid=%d", proc.pid)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        logger.exception(
            "Failed to terminate SGLang process group; trying proc.terminate()"
        )
        proc.terminate()

    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        logger.warning(
            "SGLang server pid=%d did not stop within %ds; killing it",
            proc.pid,
            grace_seconds,
        )

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception:
        logger.exception("Failed to kill SGLang process group; trying proc.kill()")
        proc.kill()
    proc.wait()


@contextmanager
def shared_sglang_server(config: Optional[SGLangBackendConfig]) -> Iterator[None]:
    """Launch one server for an orchestration scope and publish its base URL."""
    if config is None:
        yield
        return

    previous_base_url = os.environ.get(MMIRAGE_SGLANG_BASE_URL)
    previous_port = config.port
    attempted_ports: set[int] = set()
    last_error: Optional[BaseException] = None

    try:
        while len(attempted_ports) < SGLANG_PORT_SEARCH_ATTEMPTS:
            port = _next_available_port(config, attempted_ports)
            config.port = port
            attempted_ports.add(port)

            proc = launch_sglang_server(config)
            try:
                wait_for_sglang_server(proc, config)
                os.environ[MMIRAGE_SGLANG_BASE_URL] = config.resolved_base_url()
                yield
                return
            except RuntimeError as exc:
                last_error = exc
                if _is_address_in_use_failure(proc, port, exc):
                    logger.warning(
                        "SGLang port %d became unavailable during startup; trying another port",
                        port,
                    )
                    continue
                raise
            finally:
                if previous_base_url is None:
                    os.environ.pop(MMIRAGE_SGLANG_BASE_URL, None)
                else:
                    os.environ[MMIRAGE_SGLANG_BASE_URL] = previous_base_url
                stop_sglang_server(proc)

        raise RuntimeError(
            "Could not find an available localhost port for the shared SGLang "
            f"server after {SGLANG_PORT_SEARCH_ATTEMPTS} attempts starting at "
            f"{previous_port or DEFAULT_SGLANG_PORT}."
        ) from last_error
    finally:
        config.port = previous_port


def _next_available_port(
    config: SGLangBackendConfig,
    attempted_ports: set[int],
) -> int:
    start_port = config.port or DEFAULT_SGLANG_PORT
    for offset in range(SGLANG_PORT_SEARCH_ATTEMPTS):
        port = start_port + offset
        if port in attempted_ports:
            continue
        if _is_port_available(port):
            if port != start_port:
                logger.info(
                    "SGLang port %d is unavailable; using port %d instead",
                    start_port,
                    port,
                )
            return port

    raise RuntimeError(
        "Could not find an available localhost port for the shared SGLang "
        f"server after checking {SGLANG_PORT_SEARCH_ATTEMPTS} ports from {start_port}."
    )


def _is_port_available(port: int) -> bool:
    """Best-effort preflight; _is_address_in_use_failure retries recover races."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _is_address_in_use_failure(
    proc: subprocess.Popen[bytes],
    port: int,
    exc: BaseException,
) -> bool:
    message = f"{exc}\n{_format_output_tail(proc)}".lower()
    return (
        "address already in use" in message
        or "port is already in use" in message
        or "errno 98" in message
    )


def _log_process_output(proc: subprocess.Popen[bytes], output_tail: Deque[str]) -> None:
    if proc.stdout is None:
        return
    for raw_line in proc.stdout:
        line = raw_line.decode("utf-8", errors="replace").rstrip()
        if line:
            output_tail.append(line)
            logger.info("[sglang-server] %s", line)


def _format_output_tail(proc: subprocess.Popen[bytes]) -> str:
    output_tail: List[str] = list(getattr(proc, "_mmirage_output_tail", ()))
    return "\n".join(output_tail) or "(no server output captured)"


def _read_json(url: str, api_key: Optional[str]) -> None:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read readiness endpoint {url}: {exc}") from exc


def _shell_join(parts: List[str]) -> str:
    try:
        import shlex

        return shlex.join(parts)
    except Exception:  # pragma: no cover
        return " ".join(parts)
