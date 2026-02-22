#!/usr/bin/env python3
"""clod-compu-stats: Terminal dashboard for system metrics, Docker, and Claude stats."""

from __future__ import annotations

import json
import socket
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psutil
import requests
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import DataTable, Footer, Header, Static

CREDENTIALS_PATH = Path.home() / ".claude" / ".credentials.json"
USAGE_API_URL = "https://api.anthropic.com/api/oauth/usage"


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


def get_system_metrics() -> dict:
    cpu = psutil.cpu_percent(interval=0)
    freq = psutil.cpu_freq()
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
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
        "procs": procs[:8],
    }


def get_docker_containers() -> list[dict] | str:
    try:
        import docker as docker_lib

        client = docker_lib.from_env()
        client.ping()
    except Exception as exc:
        return f"Docker unavailable: {exc}"

    containers = []
    for c in client.containers.list(all=True):
        ports = ", ".join(
            f"{v[0]['HostPort']}->{k}" if v else k
            for k, v in (c.ports or {}).items()
        )
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
    # Interface addresses
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

    # Connection counts
    try:
        conns = psutil.net_connections(kind="inet")
        tcp_established = sum(1 for c in conns if c.status == "ESTABLISHED")
        tcp_listen = sum(1 for c in conns if c.status == "LISTEN")
        total_conns = len(conns)
    except (psutil.AccessDenied, OSError):
        tcp_established = tcp_listen = total_conns = 0

    # Gateway
    gateway = ""
    try:
        out = subprocess.check_output(["ip", "route", "show", "default"], text=True, timeout=5).strip()
        parts = out.split()
        if "via" in parts:
            gateway = parts[parts.index("via") + 1]
    except Exception:
        pass

    # DNS
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


def get_claude_usage_api() -> dict | str:
    """Fetch live usage limits from the Anthropic OAuth API."""
    if not CREDENTIALS_PATH.exists():
        return "Credentials not found at ~/.claude/.credentials.json"
    try:
        creds = json.loads(CREDENTIALS_PATH.read_text())
        token = creds["claudeAiOauth"]["accessToken"]
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        return f"Error reading credentials: {exc}"

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
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return f"API error: {exc}"


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

class SystemPanel(Vertical):
    def compose(self) -> ComposeResult:
        yield Static("Loading...", id="sys-summary", classes="summary-line")
        yield DataTable(id="sys-procs")

    def on_mount(self) -> None:
        psutil.cpu_percent(interval=0)
        table = self.query_one("#sys-procs", DataTable)
        table.add_columns("PID", "Name", "CPU%", "Mem%")
        self.refresh_data()
        self.set_interval(2, self.refresh_data)

    def refresh_data(self) -> None:
        m = get_system_metrics()
        summary = (
            f"CPU: {m['cpu']:.1f}% @ {m['cpu_freq_ghz']:.2f}GHz  "
            f"({m['cpu_cores_logical']} cores)  |  "
            f"Mem: {format_bytes(m['mem_used'])}/{format_bytes(m['mem_total'])} ({m['mem_pct']:.1f}%)  |  "
            f"Disk: {format_bytes(m['disk_used'])}/{format_bytes(m['disk_total'])} ({m['disk_pct']:.1f}%)"
        )
        self.query_one("#sys-summary", Static).update(summary)

        table = self.query_one("#sys-procs", DataTable)
        table.clear()
        for p in m["procs"]:
            table.add_row(
                str(p.get("pid", "")),
                (p.get("name") or "?")[:30],
                f"{p.get('cpu_percent', 0):.1f}",
                f"{p.get('memory_percent', 0):.1f}",
            )


class DockerPanel(Vertical):
    def compose(self) -> ComposeResult:
        yield DataTable(id="docker-table")

    def on_mount(self) -> None:
        table = self.query_one("#docker-table", DataTable)
        table.add_columns("Status", "Name", "Image", "Ports", "Uptime")
        self.refresh_data()
        self.set_interval(5, self.refresh_data)

    def refresh_data(self) -> None:
        table = self.query_one("#docker-table", DataTable)
        table.clear()
        result = get_docker_containers()
        if isinstance(result, str):
            table.add_row(Text("!", style="bold red"), result, "", "", "")
            return
        if not result:
            table.add_row(Text("-", style="dim"), "No containers", "", "", "")
            return
        for c in result:
            if c["status"] == "running":
                status = Text("●", style="bold green")
            else:
                status = Text("●", style="bold red")
            table.add_row(
                status,
                c["name"][:30],
                c["image"][:35],
                c["ports"][:30],
                c["uptime"],
            )


class NetworkPanel(Vertical):
    _prev_io: dict[str, tuple[int, int]] = {}

    def compose(self) -> ComposeResult:
        yield Static("Loading...", id="net-summary", classes="summary-line")
        yield DataTable(id="net-table")

    def on_mount(self) -> None:
        table = self.query_one("#net-table", DataTable)
        table.add_columns("Interface", "IP", "Rx/s", "Tx/s", "Total Rx", "Total Tx")
        # Prime IO counters for rate calculation
        info = get_network_info()
        for iface in info["ifaces"]:
            self._prev_io[iface["name"]] = (iface["recv"], iface["sent"])
        self.refresh_data()
        self.set_interval(3, self.refresh_data)

    def refresh_data(self) -> None:
        info = get_network_info()

        # Summary line
        conns = info["total_conns"]
        estab = info["tcp_established"]
        listen = info["tcp_listen"]
        gw = info["gateway"] or "N/A"
        dns = ", ".join(info["dns"]) or "N/A"
        self.query_one("#net-summary", Static).update(
            f"Connections: {conns} (established: {estab}, listen: {listen})  |  "
            f"Gateway: {gw}  |  DNS: {dns}"
        )

        # Interface table with rates
        table = self.query_one("#net-table", DataTable)
        table.clear()
        for iface in info["ifaces"]:
            name = iface["name"]
            prev_recv, prev_sent = self._prev_io.get(name, (iface["recv"], iface["sent"]))
            rx_rate = max(0, iface["recv"] - prev_recv) / 3  # 3s interval
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


class ClaudePanel(VerticalScroll):
    def compose(self) -> ComposeResult:
        yield Static("Loading...", id="claude-stats")

    def on_mount(self) -> None:
        self.refresh_data()
        self.set_interval(30, self.refresh_data)

    def refresh_data(self) -> None:
        data = get_claude_stats()
        api = get_claude_usage_api()
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

            # Per-model 7d breakdowns
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

class ClawdDashboard(App):
    CSS_PATH = "dashboard.tcss"
    TITLE = "clod-compu-stats"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh_all", "Refresh All"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield SystemPanel(id="system-panel")
        yield NetworkPanel(id="network-panel")
        yield DockerPanel(id="docker-panel")
        yield ClaudePanel(id="claude-panel")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#system-panel").border_title = "System Monitor"
        self.query_one("#network-panel").border_title = "Network"
        self.query_one("#docker-panel").border_title = "Docker Containers"
        self.query_one("#claude-panel").border_title = "Claude Usage"

    def action_refresh_all(self) -> None:
        self.query_one(SystemPanel).refresh_data()
        self.query_one(NetworkPanel).refresh_data()
        self.query_one(DockerPanel).refresh_data()
        self.query_one(ClaudePanel).refresh_data()


if __name__ == "__main__":
    ClawdDashboard().run()
