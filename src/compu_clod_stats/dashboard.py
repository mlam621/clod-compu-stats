#!/usr/bin/env python3
"""compu-clod-stats: Terminal dashboard for system metrics, Docker, and Claude stats."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
import urllib.request
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psutil
import requests
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, Static
from textual import work

CREDENTIALS_PATH = Path.home() / ".claude" / ".credentials.json"
USAGE_API_URL = "https://api.anthropic.com/api/oauth/usage"
TOKEN_REFRESH_URL = "https://platform.claude.com/v1/oauth/token"
OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
CONFIG_DIR = Path.home() / ".config" / "compu-clod-stats"
LAYOUT_PATH = CONFIG_DIR / "layout.json"


# ---------------------------------------------------------------------------
# Layout persistence
# ---------------------------------------------------------------------------

def _load_layout() -> dict:
    """Load saved panel order and collapsed state."""
    try:
        return json.loads(LAYOUT_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_layout(order: list[str], collapsed: dict[str, bool]) -> None:
    """Save panel order and collapsed state to disk."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        LAYOUT_PATH.write_text(json.dumps({"order": order, "collapsed": collapsed}, indent=2))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def format_bytes(b: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024:
            return f"{b:.1f}{unit}"
        b /= 1024
    return f"{b:.1f}PB"


def format_tokens(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def format_duration_ms(ms: int) -> str:
    secs = ms // 1000
    hours, remainder = divmod(secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def _time_until(iso_ts: str, show_days: bool = False) -> str:
    """Return human-readable time until an ISO timestamp."""
    try:
        target = datetime.fromisoformat(iso_ts)
        delta = target - datetime.now(timezone.utc)
        total_secs = int(delta.total_seconds())
        if total_secs <= 0:
            return "now"
        if show_days:
            days, rem = divmod(total_secs, 86400)
            hrs, rem = divmod(rem, 3600)
            mins, _ = divmod(rem, 60)
            if days:
                return f"{days}d {hrs}h {mins}m"
        else:
            hrs, rem = divmod(total_secs, 3600)
            mins, _ = divmod(rem, 60)
        if hrs:
            return f"{hrs}h {mins}m"
        return f"{mins}m"
    except (ValueError, TypeError):
        return "?"


def _make_bar(pct: float, width: int = 30) -> str:
    """Build a Rich-markup progress bar."""
    filled = int(pct / 100 * width)
    filled = max(0, min(filled, width))
    empty = width - filled
    if pct >= 90:
        color = "red"
    elif pct >= 70:
        color = "yellow"
    else:
        color = "green"
    return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim]"


def _format_uptime() -> str:
    """Format system uptime from boot_time."""
    boot = psutil.boot_time()
    delta = time.time() - boot
    days, rem = divmod(int(delta), 86400)
    hours, rem = divmod(rem, 3600)
    mins, _ = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {mins}m"
    if hours:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def _count_ssh_sessions() -> int:
    """Count active SSH sessions by looking for sshd child processes."""
    count = 0
    for p in psutil.process_iter(["name", "ppid"]):
        try:
            if p.info["name"] == "sshd" and p.info["ppid"] != 1:
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return count


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

def get_system_metrics() -> dict:
    cpu = psutil.cpu_percent(interval=1)
    freq = psutil.cpu_freq()
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    load = psutil.getloadavg()
    procs = []
    for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
        try:
            info = p.info
            procs.append(info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    procs.sort(key=lambda x: x.get("cpu_percent") or 0, reverse=True)
    return {
        "cpu": cpu,
        "cpu_freq_ghz": freq.current / 1000 if freq else 0,
        "cpu_cores_phys": psutil.cpu_count(logical=False) or 0,
        "cpu_cores_logical": psutil.cpu_count(logical=True) or 0,
        "mem_used": mem.used,
        "mem_total": mem.total,
        "mem_pct": mem.percent,
        "disk_used": disk.used,
        "disk_total": disk.total,
        "disk_pct": disk.percent,
        "load_1": load[0],
        "load_5": load[1],
        "load_15": load[2],
        "uptime": _format_uptime(),
        "ssh_sessions": _count_ssh_sessions(),
        "procs": procs[:8],
    }


_docker_client = None


def _get_docker_client():
    """Get or create a reusable Docker client."""
    global _docker_client
    import docker as docker_lib
    if _docker_client is None:
        _docker_client = docker_lib.from_env()
    _docker_client.ping()
    return _docker_client


def get_docker_containers() -> list[dict] | str:
    try:
        client = _get_docker_client()
    except Exception as exc:
        return f"Docker unavailable: {exc}"

    containers = []
    for c in client.containers.list(all=True):
        # Deduplicate port bindings (IPv4 + IPv6 create duplicates)
        seen_ports = set()
        port_strs = []
        for k, v in (c.ports or {}).items():
            if v:
                for binding in v:
                    hp = binding.get("HostPort", "")
                    mapping = f"{hp}->{k}"
                    if mapping not in seen_ports:
                        seen_ports.add(mapping)
                        port_strs.append(mapping)
            else:
                port_strs.append(k)
        ports = ", ".join(port_strs)
        started = c.attrs.get("State", {}).get("StartedAt", "")
        uptime = ""
        if started and c.status == "running":
            try:
                start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                delta = datetime.now(timezone.utc) - start_dt
                hours, rem = divmod(int(delta.total_seconds()), 3600)
                mins, _ = divmod(rem, 60)
                uptime = f"{hours}h {mins}m" if hours else f"{mins}m"
            except (ValueError, TypeError):
                pass
        containers.append({
            "status": c.status,
            "name": c.name,
            "image": ",".join(c.image.tags) if c.image.tags else c.image.short_id,
            "ports": ports,
            "uptime": uptime,
        })
    return containers



def get_network_info() -> dict:
    """Collect network interfaces, IPs, IO counters, connections, gateway, DNS."""
    ifaces = []
    addrs = psutil.net_if_addrs()
    io = psutil.net_io_counters(pernic=True)
    for name, addr_list in addrs.items():
        ipv4 = ""
        for a in addr_list:
            if a.family == socket.AF_INET:
                ipv4 = a.address
                break
        counters = io.get(name)
        ifaces.append({
            "name": name,
            "ip": ipv4,
            "sent": counters.bytes_sent if counters else 0,
            "recv": counters.bytes_recv if counters else 0,
        })

    try:
        conns = psutil.net_connections(kind="inet")
        tcp_established = sum(1 for c in conns if c.status == "ESTABLISHED")
        tcp_listen = sum(1 for c in conns if c.status == "LISTEN")
        total_conns = len(conns)
    except (psutil.AccessDenied, OSError):
        tcp_established = tcp_listen = total_conns = 0

    gateway = ""
    try:
        out = subprocess.check_output(["ip", "route", "show", "default"], text=True, timeout=5).strip()
        parts = out.split()
        if "via" in parts:
            gateway = parts[parts.index("via") + 1]
    except Exception:
        pass

    dns_servers = []
    try:
        for line in Path("/etc/resolv.conf").read_text().splitlines():
            if line.strip().startswith("nameserver"):
                dns_servers.append(line.strip().split()[1])
    except OSError:
        pass

    ifaces.sort(key=lambda x: (0 if x["name"].startswith("eth") else 1, x["name"]))

    return {
        "ifaces": ifaces,
        "tcp_established": tcp_established,
        "tcp_listen": tcp_listen,
        "total_conns": total_conns,
        "gateway": gateway,
        "dns": dns_servers,
    }


def get_claude_stats() -> dict | str:
    path = Path.home() / ".claude" / "stats-cache.json"
    if not path.exists():
        return "Stats file not found at ~/.claude/stats-cache.json"
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return f"Error reading stats: {exc}"
    return data


def _refresh_oauth_token() -> str | None:
    """Refresh the OAuth access token using the refresh token.

    Returns the new access token, or None on failure.
    """
    try:
        creds = json.loads(CREDENTIALS_PATH.read_text())
        refresh_token = creds["claudeAiOauth"]["refreshToken"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None

    try:
        resp = requests.post(
            TOKEN_REFRESH_URL,
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": OAUTH_CLIENT_ID,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    new_access = data.get("access_token")
    new_refresh = data.get("refresh_token")
    expires_in = data.get("expires_in", 28800)
    if not new_access:
        return None

    # Update credentials file so the token persists across restarts
    try:
        creds = json.loads(CREDENTIALS_PATH.read_text())
        creds["claudeAiOauth"]["accessToken"] = new_access
        if new_refresh:
            creds["claudeAiOauth"]["refreshToken"] = new_refresh
        creds["claudeAiOauth"]["expiresAt"] = int(time.time() * 1000) + expires_in * 1000
        CREDENTIALS_PATH.write_text(json.dumps(creds, indent=2))
    except (json.JSONDecodeError, OSError):
        pass  # Token still usable even if we can't persist it

    return new_access


def _get_oauth_token() -> str | None:
    """Read the access token, refreshing if expired or about to expire."""
    try:
        creds = json.loads(CREDENTIALS_PATH.read_text())
        oauth = creds["claudeAiOauth"]
        token = oauth["accessToken"]
        expires_at = oauth.get("expiresAt", 0)
    except (json.JSONDecodeError, KeyError, OSError):
        return None

    # Refresh if token expires within 5 minutes
    if expires_at < (time.time() * 1000) + 300_000:
        return _refresh_oauth_token() or token

    return token


def get_claude_usage_api() -> dict | str:
    """Fetch live usage limits from the Anthropic OAuth API."""
    if not CREDENTIALS_PATH.exists():
        return "Credentials not found at ~/.claude/.credentials.json"

    token = _get_oauth_token()
    if not token:
        return "Error reading credentials"

    try:
        resp = requests.get(
            USAGE_API_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "anthropic-beta": "oauth-2025-04-20",
            },
            timeout=10,
        )
        # On 401, try refreshing the token once and retry
        if resp.status_code == 401:
            new_token = _refresh_oauth_token()
            if new_token:
                resp = requests.get(
                    USAGE_API_URL,
                    headers={
                        "Authorization": f"Bearer {new_token}",
                        "Content-Type": "application/json",
                        "anthropic-beta": "oauth-2025-04-20",
                    },
                    timeout=10,
                )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return f"API error: {exc}"


def get_git_repos() -> list[dict]:
    """Scan ~/ recursively for git repos, skipping hidden/non-project dirs."""
    repos = []
    home = Path.home()
    _skip_dirs = {
        ".cache", ".local", ".config", ".claude", ".npm", ".nvm", ".cargo",
        ".rustup", ".pyenv", ".venv", "venv", "__pycache__", "node_modules",
        ".docker", ".snap", ".ssh", ".gnupg", ".mozilla", ".vscode-server",
    }

    repo_dirs: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(home):
        # Prune dirs we don't want to descend into
        dirnames[:] = [
            d for d in dirnames
            if d not in _skip_dirs and not (d.startswith(".") and d != ".git")
        ]
        if ".git" in dirnames and Path(dirpath, ".git").is_dir():
            repo_dirs.append(Path(dirpath))
            dirnames.remove(".git")  # don't descend into .git itself

    for d in sorted(repo_dirs):
        rel = d.relative_to(home)
        repo = {"name": str(rel), "branch": "?", "changes": 0, "last_commit": "?"}
        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(d), text=True, timeout=5, stderr=subprocess.DEVNULL,
            ).strip()
            repo["branch"] = branch
        except Exception:
            pass
        try:
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=str(d), text=True, timeout=5, stderr=subprocess.DEVNULL,
            ).strip()
            lines = [
                l for l in status.splitlines()
                if not l[3:].startswith(".claude/")
            ]
            repo["changes"] = len(lines)
        except Exception:
            pass
        try:
            log = subprocess.check_output(
                ["git", "log", "-1", "--format=%h %s"],
                cwd=str(d), text=True, timeout=5, stderr=subprocess.DEVNULL,
            ).strip()
            repo["last_commit"] = log[:60]
        except Exception:
            pass
        repos.append(repo)
    return repos


def _check_single_container(name: str, host_port: str) -> dict:
    """Check health of a single container port (runs in thread pool)."""
    for path in ["/api/health", "/health", "/"]:
        url = f"http://127.0.0.1:{host_port}{path}"
        try:
            start = time.time()
            req = urllib.request.urlopen(url, timeout=3)
            elapsed = (time.time() - start) * 1000
            status_code = req.getcode()
            return {
                "container": name,
                "url": f":{host_port}{path}",
                "status": "OK" if 200 <= status_code < 400 else f"{status_code}",
                "response_ms": f"{elapsed:.0f}ms",
                "ok": 200 <= status_code < 400,
            }
        except Exception:
            continue
    return {
        "container": name,
        "url": f":{host_port}",
        "status": "FAIL",
        "response_ms": "-",
        "ok": False,
    }


def get_health_checks() -> list[dict]:
    """Check health endpoints on running Docker containers with exposed ports."""
    try:
        client = _get_docker_client()
    except Exception:
        return []

    # Collect targets
    targets: list[tuple[str, str]] = []
    for c in client.containers.list():
        if not c.ports:
            continue
        for port_key, bindings in (c.ports or {}).items():
            if not bindings:
                continue
            host_port = bindings[0].get("HostPort")
            if not host_port:
                continue
            targets.append((c.name, host_port))
            break  # Only check first exposed port per container

    if not targets:
        return []

    # Check all containers in parallel
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(len(targets), 10)) as pool:
        futures = {pool.submit(_check_single_container, name, port): name for name, port in targets}
        for future in as_completed(futures, timeout=10):
            try:
                results.append(future.result())
            except Exception:
                results.append({
                    "container": futures[future],
                    "url": "?",
                    "status": "FAIL",
                    "response_ms": "-",
                    "ok": False,
                })
    results.sort(key=lambda r: r["container"])
    return results


# ---------------------------------------------------------------------------
# CollapsiblePanel base class
# ---------------------------------------------------------------------------

class CollapsiblePanel(Vertical, can_focus=True):
    """Base class for all dashboard panels with collapse/expand support."""

    PANEL_TITLE: str = "Panel"
    REFRESH_INTERVAL: float = 5.0
    EXPANDED_HEIGHT: str = "auto"

    collapsed: reactive[bool] = reactive(False)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._refresh_timer = None
        self._last_data: object = None  # Cache for last fetched data

    def compose(self) -> ComposeResult:
        yield Static("Loading...", id=f"{self.id}-summary", classes="summary-line")
        yield from self.compose_expanded()

    @abstractmethod
    def compose_expanded(self) -> ComposeResult:
        """Yield widgets shown only when expanded."""
        ...

    @abstractmethod
    def get_summary(self) -> str:
        """Return Rich text for the collapsed summary line (should use _last_data)."""
        ...

    @abstractmethod
    def refresh_data(self) -> None:
        """Fetch data and update widgets (should use @work(thread=True))."""
        ...

    def _tick_refresh(self) -> None:
        """Timer callback: always refresh data to keep cache and summary current."""
        self.refresh_data()

    def on_mount(self) -> None:
        self.border_title = self.PANEL_TITLE
        self._setup_columns()
        self.refresh_data()
        self._refresh_timer = self.set_interval(self.REFRESH_INTERVAL, self._tick_refresh)

    def _setup_columns(self) -> None:
        """Override in subclasses that need to set up DataTable columns."""
        pass

    def watch_collapsed(self, value: bool) -> None:
        """Toggle visibility of expanded widgets and manage timer."""
        summary = self.query_one(f"#{self.id}-summary", Static)

        if value:
            # Collapse: hide expanded widgets, show summary
            summary.update(self.get_summary())
            for child in self.children:
                if child is not summary:
                    child.display = False
            self.add_class("collapsed")
            self.styles.height = "auto"
        else:
            # Expand: show expanded widgets
            for child in self.children:
                child.display = True
            self.remove_class("collapsed")
            self.styles.height = self.EXPANDED_HEIGHT

    def toggle(self) -> None:
        self.collapsed = not self.collapsed


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

class SystemPanel(CollapsiblePanel):
    PANEL_TITLE = "System Monitor"
    REFRESH_INTERVAL = 2.0

    def compose_expanded(self) -> ComposeResult:
        yield Static("", id="sys-extra", classes="summary-line")
        yield DataTable(id="sys-procs")

    def _setup_columns(self) -> None:
        table = self.query_one("#sys-procs", DataTable)
        table.add_columns("PID", "Name", "CPU%", "Mem%")

    def _available_width(self) -> int:
        """Return usable content width (accounting for border + padding)."""
        return self.size.width - 4 if self.size.width > 4 else 40

    def _format_summary(self, m: dict) -> str:
        """Format CPU/RAM/Disk summary based on available width."""
        w = self._available_width()
        cpu_full = f"CPU: {m['cpu']:.1f}% @ {m['cpu_freq_ghz']:.2f}GHz  ({m['cpu_cores_logical']}c)"
        ram_full = f"RAM: {m['mem_pct']:.1f}%  ({format_bytes(m['mem_used'])}/{format_bytes(m['mem_total'])})"
        disk_full = f"Disk: {format_bytes(m['disk_used'])}/{format_bytes(m['disk_total'])} ({m['disk_pct']:.1f}%)"
        wide = f"{cpu_full}  |  {ram_full}  |  {disk_full}"
        if w >= len(wide):
            return wide
        medium = f"{cpu_full}\n{ram_full}\n{disk_full}"
        if w >= max(len(cpu_full), len(ram_full), len(disk_full)):
            return medium
        # Narrow: percentages only
        return f"CPU: {m['cpu']:.0f}%  |  RAM: {m['mem_pct']:.0f}%  |  Disk: {m['disk_pct']:.0f}%"

    def _format_extra(self, m: dict) -> str:
        """Format Load/Uptime/SSH based on available width."""
        w = self._available_width()
        load = f"Load: {m['load_1']:.2f} / {m['load_5']:.2f} / {m['load_15']:.2f}"
        uptime = f"Uptime: {m['uptime']}"
        ssh = f"SSH sessions: {m['ssh_sessions']}"
        wide = f"{load}  |  {uptime}  |  {ssh}"
        if w >= len(wide):
            return wide
        return f"{load}\n{uptime}\n{ssh}"

    def get_summary(self) -> str:
        m = self._last_data
        if not m:
            return "CPU: ? | RAM: ? | Disk: ?"
        w = self._available_width()
        full = (
            f"CPU: {m['cpu']:.1f}%  |  RAM: {m['mem_pct']:.1f}%  |  "
            f"Disk: {m['disk_pct']:.1f}%  |  Up: {m['uptime']}  |  SSH: {m['ssh_sessions']}"
        )
        if w >= len(full):
            return full
        return f"CPU: {m['cpu']:.0f}%  |  RAM: {m['mem_pct']:.0f}%  |  Disk: {m['disk_pct']:.0f}%"

    def refresh_data(self) -> None:
        self._do_refresh()

    @work(thread=True)
    def _do_refresh(self) -> None:
        m = get_system_metrics()
        self.app.call_from_thread(self._update_ui, m)

    def _update_ui(self, m: dict) -> None:
        self._last_data = m
        self.query_one(f"#{self.id}-summary", Static).update(self._format_summary(m))
        self.query_one("#sys-extra", Static).update(self._format_extra(m))

    def on_resize(self) -> None:
        """Re-render with current data when terminal is resized."""
        m = self._last_data
        if not m:
            return
        self.query_one(f"#{self.id}-summary", Static).update(
            self.get_summary() if self.collapsed else self._format_summary(m)
        )
        if not self.collapsed:
            self.query_one("#sys-extra", Static).update(self._format_extra(m))

        table = self.query_one("#sys-procs", DataTable)
        table.clear()
        for p in m["procs"]:
            table.add_row(
                str(p.get("pid", "")),
                (p.get("name") or "?")[:30],
                f"{p.get('cpu_percent', 0):.1f}",
                f"{p.get('memory_percent', 0):.1f}",
            )


class NetworkPanel(CollapsiblePanel):
    PANEL_TITLE = "Network"
    REFRESH_INTERVAL = 3.0

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._prev_io: dict[str, tuple[int, int]] = {}

    def compose_expanded(self) -> ComposeResult:
        yield DataTable(id="net-table")

    def _setup_columns(self) -> None:
        table = self.query_one("#net-table", DataTable)
        table.add_columns("Interface", "IP", "Rx/s", "Tx/s", "Total Rx", "Total Tx")

    def on_mount(self) -> None:
        self.border_title = self.PANEL_TITLE
        self._setup_columns()
        self._prime_io_counters()
        self._refresh_timer = self.set_interval(self.REFRESH_INTERVAL, self._tick_refresh)

    @work(thread=True)
    def _prime_io_counters(self) -> None:
        info = get_network_info()
        self.app.call_from_thread(self._store_initial_io, info)

    def _store_initial_io(self, info: dict) -> None:
        self._last_data = info
        for iface in info["ifaces"]:
            self._prev_io[iface["name"]] = (iface["recv"], iface["sent"])

    def get_summary(self) -> str:
        info = self._last_data
        if not info:
            return "Connections: ? | Gateway: ? | DNS: ?"
        gw = info["gateway"] or "N/A"
        dns = ", ".join(info["dns"]) or "N/A"
        return f"Connections: {info['total_conns']} | Gateway: {gw} | DNS: {dns}"

    def watch_collapsed(self, value: bool) -> None:
        super().watch_collapsed(value)
        if not value:
            # Re-prime IO counters to avoid stale rate spike on expand
            self._prime_io_counters()

    def refresh_data(self) -> None:
        self._do_refresh()

    @work(thread=True)
    def _do_refresh(self) -> None:
        info = get_network_info()
        self.app.call_from_thread(self._update_ui, info)

    def _update_ui(self, info: dict) -> None:
        self._last_data = info

        conns = info["total_conns"]
        estab = info["tcp_established"]
        listen = info["tcp_listen"]
        gw = info["gateway"] or "N/A"
        dns = ", ".join(info["dns"]) or "N/A"
        self.query_one(f"#{self.id}-summary", Static).update(
            f"Connections: {conns} (established: {estab}, listen: {listen})  |  "
            f"Gateway: {gw}  |  DNS: {dns}"
        )

        table = self.query_one("#net-table", DataTable)
        table.clear()
        for iface in info["ifaces"]:
            name = iface["name"]
            prev_recv, prev_sent = self._prev_io.get(name, (iface["recv"], iface["sent"]))
            rx_rate = max(0, iface["recv"] - prev_recv) / 3
            tx_rate = max(0, iface["sent"] - prev_sent) / 3
            self._prev_io[name] = (iface["recv"], iface["sent"])

            table.add_row(
                name,
                iface["ip"] or "-",
                f"{format_bytes(rx_rate)}/s",
                f"{format_bytes(tx_rate)}/s",
                format_bytes(iface["recv"]),
                format_bytes(iface["sent"]),
            )


class DockerPanel(CollapsiblePanel):
    PANEL_TITLE = "Docker Containers"
    REFRESH_INTERVAL = 5.0

    def compose_expanded(self) -> ComposeResult:
        yield DataTable(id="docker-table")

    def _setup_columns(self) -> None:
        table = self.query_one("#docker-table", DataTable)
        table.add_columns("Status", "Name", "Image", "Ports", "Uptime")

    def get_summary(self) -> str:
        result = self._last_data
        if result is None:
            return "Docker: ?"
        if isinstance(result, str):
            return result
        if not result:
            return "No containers"
        running = sum(1 for c in result if c["status"] == "running")
        stopped = len(result) - running
        return f"{running} running, {stopped} stopped"

    def refresh_data(self) -> None:
        self._do_refresh()

    @work(thread=True)
    def _do_refresh(self) -> None:
        result = get_docker_containers()
        self.app.call_from_thread(self._update_ui, result)

    def _update_ui(self, result: list[dict] | str) -> None:
        self._last_data = result
        # Update summary line
        if isinstance(result, str):
            self.query_one(f"#{self.id}-summary", Static).update(result)
        elif not result:
            self.query_one(f"#{self.id}-summary", Static).update("No containers")
        else:
            running = sum(1 for c in result if c["status"] == "running")
            stopped = len(result) - running
            self.query_one(f"#{self.id}-summary", Static).update(
                f"{running} running, {stopped} stopped"
            )

        table = self.query_one("#docker-table", DataTable)
        table.clear()
        if isinstance(result, str):
            table.add_row(Text("!", style="bold red"), result, "", "", "")
        elif not result:
            table.add_row(Text("-", style="dim"), "No containers", "", "", "")
        else:
            for c in result:
                if c["status"] == "running":
                    status = Text("●", style="bold green")
                else:
                    status = Text("●", style="bold red")
                table.add_row(
                    status,
                    c["name"][:30],
                    c["image"][:35],
                    c["ports"][:40],
                    c["uptime"],
                )


class GitPanel(CollapsiblePanel):
    PANEL_TITLE = "Git Repositories"
    REFRESH_INTERVAL = 30.0

    def compose_expanded(self) -> ComposeResult:
        yield DataTable(id="git-table")

    def _setup_columns(self) -> None:
        table = self.query_one("#git-table", DataTable)
        table.add_columns("Repo", "Branch", "Changes", "Last Commit")

    def get_summary(self) -> str:
        repos = self._last_data
        if repos is None:
            return "Git: ?"
        dirty = sum(1 for r in repos if r["changes"] > 0)
        return f"{len(repos)} repos, {dirty} with uncommitted changes"

    def refresh_data(self) -> None:
        self._do_refresh()

    @work(thread=True)
    def _do_refresh(self) -> None:
        repos = get_git_repos()
        self.app.call_from_thread(self._update_ui, repos)

    def _update_ui(self, repos: list[dict]) -> None:
        self._last_data = repos
        dirty = sum(1 for r in repos if r["changes"] > 0)
        self.query_one(f"#{self.id}-summary", Static).update(
            f"{len(repos)} repos, {dirty} with uncommitted changes"
        )

        table = self.query_one("#git-table", DataTable)
        table.clear()
        if not repos:
            table.add_row("No repos found", "", "", "")
            return
        for r in repos:
            changes_text = str(r["changes"])
            if r["changes"] > 0:
                changes = Text(changes_text, style="bold red")
            else:
                changes = Text(changes_text, style="green")
            table.add_row(
                r["name"][:20],
                r["branch"][:20],
                changes,
                r["last_commit"][:50],
            )


class HealthPanel(CollapsiblePanel):
    PANEL_TITLE = "Health Checks"
    REFRESH_INTERVAL = 15.0

    def compose_expanded(self) -> ComposeResult:
        yield DataTable(id="health-table")

    def _setup_columns(self) -> None:
        table = self.query_one("#health-table", DataTable)
        table.add_columns("Container", "URL", "Status", "Response")

    def get_summary(self) -> str:
        checks = self._last_data
        if checks is None:
            return "Health: ?"
        if not checks:
            return "No health endpoints found"
        healthy = sum(1 for c in checks if c["ok"])
        return f"{healthy}/{len(checks)} healthy"

    @work(thread=True)
    def _do_health_check(self) -> None:
        checks = get_health_checks()
        self.app.call_from_thread(self._update_table, checks)

    def _update_table(self, checks: list[dict]) -> None:
        self._last_data = checks
        if not checks:
            self.query_one(f"#{self.id}-summary", Static).update("No health endpoints found")
        else:
            healthy = sum(1 for c in checks if c["ok"])
            self.query_one(f"#{self.id}-summary", Static).update(f"{healthy}/{len(checks)} healthy")

        table = self.query_one("#health-table", DataTable)
        table.clear()
        if not checks:
            table.add_row("No endpoints", "", "", "")
            return
        for c in checks:
            if c["ok"]:
                status = Text(c["status"], style="bold green")
            else:
                status = Text(c["status"], style="bold red")
            table.add_row(
                c["container"][:25],
                c["url"],
                status,
                c["response_ms"],
            )

    def refresh_data(self) -> None:
        self._do_health_check()


class ClaudePanel(CollapsiblePanel):
    PANEL_TITLE = "Claude Usage"
    REFRESH_INTERVAL = 30.0

    def compose_expanded(self) -> ComposeResult:
        yield VerticalScroll(Static("Loading...", id="claude-stats"), id="claude-scroll")

    def get_summary(self) -> str:
        cached = self._last_data
        if not cached or not isinstance(cached, dict):
            return "Claude: ?"
        api = cached.get("api")
        if isinstance(api, dict):
            five_h = api.get("five_hour", {})
            seven_d = api.get("seven_day", {})
            pct_5h = five_h.get("utilization", 0) if five_h else 0
            pct_7d = seven_d.get("utilization", 0) if seven_d else 0
            return f"5h: {pct_5h:.0f}% | 7d: {pct_7d:.0f}%"
        return "Claude Usage"

    def refresh_data(self) -> None:
        self._do_refresh()

    @work(thread=True)
    def _do_refresh(self) -> None:
        data = get_claude_stats()
        api = get_claude_usage_api()
        self.app.call_from_thread(self._update_ui, data, api)

    def _update_ui(self, data: dict | str, api: dict | str) -> None:
        self._last_data = {"data": data, "api": api}

        # Update summary line
        if isinstance(api, dict):
            five_h = api.get("five_hour", {})
            seven_d = api.get("seven_day", {})
            pct_5h = five_h.get("utilization", 0) if five_h else 0
            pct_7d = seven_d.get("utilization", 0) if seven_d else 0
            self.query_one(f"#{self.id}-summary", Static).update(
                f"5h: {pct_5h:.0f}% | 7d: {pct_7d:.0f}%"
            )
        else:
            self.query_one(f"#{self.id}-summary", Static).update("Claude Usage")

        widget = self.query_one("#claude-stats", Static)

        lines: list[str] = []

        # ── Live usage from API ──
        if isinstance(api, dict):
            five_h = api.get("five_hour") or {}
            seven_d = api.get("seven_day") or {}
            extra = api.get("extra_usage") or {}

            if five_h:
                pct = five_h.get("utilization", 0)
                resets = five_h.get("resets_at", "")
                bar = _make_bar(pct)
                lines.append(f"[bold]5h Window:[/bold]  {bar} {pct:.0f}%    [dim]resets in {_time_until(resets)}[/dim]")

            if seven_d:
                pct = seven_d.get("utilization", 0)
                resets = seven_d.get("resets_at", "")
                bar = _make_bar(pct)
                lines.append(f"[bold]7d Window:[/bold]  {bar} {pct:.0f}%    [dim]resets in {_time_until(resets, show_days=True)}[/dim]")

            for key, label in [
                ("seven_day_opus", "7d Opus"),
                ("seven_day_sonnet", "7d Sonnet"),
            ]:
                bucket = api.get(key)
                if bucket:
                    pct = bucket.get("utilization", 0)
                    resets = bucket.get("resets_at", "")
                    bar = _make_bar(pct)
                    lines.append(f"[bold]{label}:[/bold]  {bar} {pct:.0f}%    [dim]resets in {_time_until(resets, show_days=True)}[/dim]")

            if extra and extra.get("is_enabled"):
                used = extra.get("used_credits", 0) / 100
                limit = extra.get("monthly_limit", 0) / 100
                eu_pct = extra.get("utilization", 0)
                bar = _make_bar(eu_pct)
                lines.append(f"[bold]Extra $:  [/bold]  {bar} {eu_pct:.1f}%    [dim]${used:.2f} / ${limit:.2f}[/dim]")

            lines.append("")
        elif isinstance(api, str):
            lines.append(f"[bold red]{api}[/bold red]")
            lines.append("[dim]Wait a minute and retry — the API may need time to sync.[/dim]")
            lines.append("")

        # ── Stats from cache file ──
        if isinstance(data, str):
            lines.append(f"[bold red]{data}[/bold red]")
            widget.update("\n".join(lines))
            return

        total_sessions = data.get("totalSessions", 0)
        total_messages = data.get("totalMessages", 0)
        lines.append(f"[bold]Sessions:[/bold] {total_sessions}    [bold]Messages:[/bold] {total_messages:,}")

        longest = data.get("longestSession")
        if longest:
            dur = format_duration_ms(longest.get("duration", 0))
            msgs = longest.get("messageCount", 0)
            lines.append(f"[bold]Longest session:[/bold] {dur} ({msgs:,} messages)")

        model_usage = data.get("modelUsage", {})
        if model_usage:
            lines.append("")
            lines.append("[bold underline]Model Token Usage[/bold underline]")
            total_cost = 0.0
            for model, usage in model_usage.items():
                inp = usage.get("inputTokens", 0)
                out = usage.get("outputTokens", 0)
                cache_read = usage.get("cacheReadInputTokens", 0)
                cache_create = usage.get("cacheCreationInputTokens", 0)
                cost = usage.get("costUSD", 0)
                total_cost += cost
                lines.append(
                    f"  [bold]{model}[/bold]  "
                    f"in={format_tokens(inp)}  out={format_tokens(out)}  "
                    f"cache_read={format_tokens(cache_read)}  cache_write={format_tokens(cache_create)}"
                )
            lines.append(f"  [bold]Total cost:[/bold] ${total_cost:.2f}")

        daily = data.get("dailyActivity", [])
        if daily:
            lines.append("")
            lines.append("[bold underline]Recent Daily Activity[/bold underline]")
            for day in daily[-7:]:
                lines.append(
                    f"  {day['date']}  "
                    f"msgs={day.get('messageCount', 0):,}  "
                    f"sessions={day.get('sessionCount', 0)}  "
                    f"tools={day.get('toolCallCount', 0):,}"
                )

        hour_counts = data.get("hourCounts", {})
        if hour_counts:
            peak_hour = max(hour_counts, key=lambda h: hour_counts[h])
            lines.append(f"\n[bold]Peak usage hour:[/bold] {int(peak_hour):02d}:00 ({hour_counts[peak_hour]} sessions)")

        widget.update("\n".join(lines))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

PANEL_REGISTRY: dict[str, type[CollapsiblePanel]] = {
    "system-panel": SystemPanel,
    "network-panel": NetworkPanel,
    "git-panel": GitPanel,
    "docker-panel": DockerPanel,
    "health-panel": HealthPanel,
    "claude-panel": ClaudePanel,
}
DEFAULT_ORDER = list(PANEL_REGISTRY.keys())


class ClawdDashboard(App):
    CSS_PATH = "dashboard.tcss"
    TITLE = f"{platform.node()} | compu-clod-stats"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh_all", "Refresh"),
        ("1", "toggle_panel(0)", "System"),
        ("2", "toggle_panel(1)", "Network"),
        ("3", "toggle_panel(2)", "Git"),
        ("4", "toggle_panel(3)", "Docker"),
        ("5", "toggle_panel(4)", "Health"),
        ("6", "toggle_panel(5)", "Claude"),
        ("ctrl+up", "move_panel_up", "Move Up"),
        ("ctrl+down", "move_panel_down", "Move Down"),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._saved_layout = _load_layout()

    def compose(self) -> ComposeResult:
        saved_order = self._saved_layout.get("order", [])
        # Use saved order, falling back to defaults for any missing/new panels
        order = [pid for pid in saved_order if pid in PANEL_REGISTRY]
        for pid in DEFAULT_ORDER:
            if pid not in order:
                order.append(pid)

        yield Header()
        with VerticalScroll(id="panel-scroll"):
            for pid in order:
                yield PANEL_REGISTRY[pid](id=pid)
        yield Footer()

    def on_mount(self) -> None:
        saved_collapsed = self._saved_layout.get("collapsed", {})
        for panel in self._get_panels():
            if saved_collapsed.get(panel.id, False):
                panel.collapsed = True

    def _get_panels(self) -> list[CollapsiblePanel]:
        """Get all panels in current DOM order."""
        return list(self.query(CollapsiblePanel))

    def _persist_layout(self) -> None:
        """Save current panel order and collapsed state to disk."""
        panels = self._get_panels()
        order = [p.id for p in panels]
        collapsed = {p.id: p.collapsed for p in panels}
        _save_layout(order, collapsed)

    def action_toggle_panel(self, index: int) -> None:
        panels = self._get_panels()
        if 0 <= index < len(panels):
            panels[index].toggle()
            self._persist_layout()

    def action_refresh_all(self) -> None:
        for panel in self._get_panels():
            if not panel.collapsed:
                panel.refresh_data()

    def action_move_panel_up(self) -> None:
        focused = self.focused
        if not isinstance(focused, CollapsiblePanel):
            return
        panels = self._get_panels()
        idx = panels.index(focused)
        if idx <= 0:
            return
        sibling = panels[idx - 1]
        container = self.query_one("#panel-scroll")
        container.move_child(focused, before=sibling)
        self._persist_layout()

    def action_move_panel_down(self) -> None:
        focused = self.focused
        if not isinstance(focused, CollapsiblePanel):
            return
        panels = self._get_panels()
        idx = panels.index(focused)
        if idx >= len(panels) - 1:
            return
        sibling = panels[idx + 1]
        container = self.query_one("#panel-scroll")
        container.move_child(focused, after=sibling)
        self._persist_layout()


def main():
    ClawdDashboard().run()


if __name__ == "__main__":
    main()
