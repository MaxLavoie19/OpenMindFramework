import argparse
import logging
import subprocess
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

from openmind.dashboard.constant.dashboard_constant import DEFAULT_PORT, DEFAULT_REFRESH_SECONDS
from openmind.dashboard.factory.dashboard_factory import create_dashboard_service
from openmind.dashboard.mapper.dashboard_html_mapper import DashboardHtmlMapper
from openmind.dashboard.model.dashboard_settings import DashboardSettings

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Serves a page following a domain's value training: the current round's progress, the machine, every round and
    the latest rules, taking a fresh snapshot for every request."""
    parser = argparse.ArgumentParser(prog="openmind-dashboard", description="Serve a page following a value training.")
    parser.add_argument("domain", help="domain whose training to follow, such as chess")
    parser.add_argument(
        "--host",
        default=None,
        help="address to listen on (default: this machine's Tailscale IPv4 address, so only the tailnet reaches it)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to listen on (default: {DEFAULT_PORT})")
    parser.add_argument(
        "--refresh", type=int, default=DEFAULT_REFRESH_SECONDS, help=f"seconds between page reloads (default: {DEFAULT_REFRESH_SECONDS})"
    )
    parser.add_argument("--report-directory", default="data/training", help="where training reports are (default: data/training)")
    parser.add_argument(
        "--log-directory", default="data/log/train-values", help="where training logs are (default: data/log/train-values)"
    )
    parser.add_argument(
        "--syslog", default="/var/log/syslog", help="system log earlyoom writes to; 'none' skips it (default: /var/log/syslog)"
    )
    parser.add_argument(
        "--dashboard-log-directory", default="data/log/dashboard", help="where the dashboard's log is saved (default: data/log/dashboard)"
    )
    arguments = parser.parse_args(argv)
    host = arguments.host or _tailscale_address()
    if host is None:
        parser.error("no Tailscale IPv4 address found: give --host")
    settings = DashboardSettings(
        arguments.domain,
        Path(arguments.report_directory),
        Path(arguments.log_directory),
        None if arguments.syslog == "none" else Path(arguments.syslog),
    )
    directory = Path(arguments.dashboard_log_directory)
    directory.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(directory / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(name)s %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    server = ThreadingHTTPServer((host, arguments.port), _handler(settings, arguments.refresh))
    logger.info("Serving the %s training on http://%s:%d", settings.domain, host, arguments.port)
    print(f"Serving the {settings.domain} training on http://{host}:{arguments.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopped")
    finally:
        server.server_close()
        root.removeHandler(handler)
        handler.close()


def page(settings: DashboardSettings, refresh: int) -> tuple[int, bytes]:
    """A status code and the page for one request, from a snapshot taken now; a snapshot that fails gives a short error
    page, logged."""
    try:
        snapshot = _SERVICE.snapshot(settings)
        return 200, DashboardHtmlMapper().to_html(snapshot, refresh).encode("utf-8")
    except Exception:
        logger.exception("The snapshot failed")
        return 500, b"<!doctype html><title>Dashboard error</title><p>The snapshot failed; see the dashboard's log.</p>"


_SERVICE = create_dashboard_service()
_LOCK = Lock()


def _handler(settings: DashboardSettings, refresh: int) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            with _LOCK:
                status, body = page(settings, refresh)
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            logger.debug("%s %s", self.address_string(), format % args)

    return DashboardHandler


def _tailscale_address() -> str | None:
    try:
        found = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=10, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    addresses = found.stdout.split()
    return addresses[0] if found.returncode == 0 and addresses else None


if __name__ == "__main__":
    main()
