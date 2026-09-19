import argparse
import logging
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

from openmind.entrypoint.debug_options import add_debug_option, start_debugging
from openmind.dashboard.constant.dashboard_constant import DEFAULT_PORT, DEFAULT_REFRESH_SECONDS
from openmind.dashboard.factory.dashboard_factory import create_dashboard_service
from openmind.dashboard.mapper.dashboard_html_mapper import DashboardHtmlMapper
from openmind.dashboard.model.dashboard_settings import DashboardSettings

logger = logging.getLogger(__name__)

#: Where a decisive game's page is served, followed by its record id.
GAME_PATH = "/game/"


def main(argv: list[str] | None = None) -> None:
    """Serves pages following a domain's value training: the training's page (its progress, the machine, every round,
    the latest rules and the latest decisive game), the list of decisive games at /games, and each decisive game at
    /game/<id>, reading afresh for every request."""
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
    parser.add_argument(
        "--log-directory", default="data/log/train-values", help="where training logs are (default: data/log/train-values)"
    )
    parser.add_argument(
        "--syslog", default="/var/log/syslog", help="system log earlyoom writes to; 'none' skips it (default: /var/log/syslog)"
    )
    parser.add_argument(
        "--dashboard-log-directory", default="data/log/dashboard", help="where the dashboard's log is saved (default: data/log/dashboard)"
    )
    add_debug_option(parser)
    arguments = parser.parse_args(argv)
    host = arguments.host or _tailscale_address()
    if host is None:
        parser.error("no Tailscale IPv4 address found: give --host")
    settings = DashboardSettings(
        arguments.domain,
        Path(arguments.log_directory),
        None if arguments.syslog == "none" else Path(arguments.syslog),
    )
    debugger = start_debugging(arguments, "dashboard", Path(arguments.dashboard_log_directory))
    server = ThreadingHTTPServer((host, arguments.port), _handler(settings, arguments.refresh))
    logger.info("Serving the %s training on http://%s:%d", settings.domain, host, arguments.port)
    print(f"Serving the {settings.domain} training on http://{host}:{arguments.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopped")
    finally:
        server.server_close()
        debugger.stop()


def page(settings: DashboardSettings, refresh: int, path: str = "/") -> tuple[int, bytes]:
    """A status code and the page a request's path asks for: `/` the training's page, from a snapshot taken now;
    `/games` the list of decisive games; `/game/<id>` one decisive game, 404 when there's no such game; anything else
    404. A page that fails gives a short error page, logged."""
    mapper = DashboardHtmlMapper()
    try:
        if path in ("/", "/index.html"):
            return 200, mapper.to_html(_SERVICE.snapshot(settings), refresh).encode("utf-8")
        browser = _SERVICE.game_browser
        if path == "/games":
            games = browser.decisive(settings.knowledge_directory, settings.domain)
            return 200, mapper.games_page(settings.domain, games, refresh).encode("utf-8")
        if path.startswith(GAME_PATH) and len(path) > len(GAME_PATH):
            game = browser.game(settings.knowledge_directory, settings.domain, path[len(GAME_PATH):])
            if game is not None:
                return 200, mapper.game_page(settings.domain, game, refresh).encode("utf-8")
            return 404, mapper.missing_page(settings.domain).encode("utf-8")
        return 404, b"<!doctype html><title>Not found</title><p>Not found.</p>"
    except Exception:
        logger.exception("The page for %s failed", path)
        return 500, b"<!doctype html><title>Dashboard error</title><p>The page failed; see the dashboard's log.</p>"


_SERVICE = create_dashboard_service()
_LOCK = Lock()


def _handler(settings: DashboardSettings, refresh: int) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            with _LOCK:
                status, body = page(settings, refresh, self.path.split("?", 1)[0])
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
