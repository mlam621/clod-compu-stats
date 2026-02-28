# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
./setup.sh        # Create .venv + install global `ccs` command via pipx
./run.sh           # Launch the dashboard (or just run `ccs` from anywhere)
```

`setup.sh` handles both the local dev `.venv` and the global `ccs` command via pipx. Re-run it after pulling changes to update both.

To install a new dependency: `.venv/bin/pip install <package>` and add it to `pyproject.toml` under `[project] dependencies`.

## Project Structure

```
src/compu_clod_stats/
├── __init__.py        # Package marker
├── __main__.py        # `python -m compu_clod_stats` entry point
├── dashboard.py       # All panels, data fetchers, and app class
└── dashboard.tcss     # Textual CSS styling
```

Entry point: `compu_clod_stats.dashboard:main` (defined in `pyproject.toml` `[project.scripts]`).

## Architecture

Textual TUI app with six collapsible, reorderable panels inside a `VerticalScroll` container. All panels extend `CollapsiblePanel(Vertical, can_focus=True)`.

### CollapsiblePanel base class

- `collapsed: reactive[bool]` — toggles between summary-only and full view
- `PANEL_TITLE`, `REFRESH_INTERVAL`, `EXPANDED_HEIGHT` — class vars per subclass
- `compose()` yields a summary `Static` + `compose_expanded()` (abstract)
- `get_summary()` (abstract) — Rich text for collapsed mode; **must read from `_last_data` cache, never re-fetch**
- `refresh_data()` (abstract) — dispatch to a `@work(thread=True)` worker; worker fetches data, then calls `app.call_from_thread()` to update UI and `_last_data`
- `_last_data: object` — cached result from the last successful fetch; used by `get_summary()` for zero-I/O summaries
- `_tick_refresh()` — always calls `refresh_data()` (even when collapsed) to keep cache and summary current
- `watch_collapsed()` — hides/shows expanded widgets, sets height to `auto` or `EXPANDED_HEIGHT`
- Timer stored in `_refresh_timer`

### Threading model

All data fetching runs off the main thread via `@work(thread=True)`. Each panel follows this pattern:
1. `refresh_data()` calls `self._do_refresh()` (the `@work` method)
2. `_do_refresh()` fetches data in a background thread
3. Calls `self.app.call_from_thread(self._update_ui, data)` to update DOM on the main thread
4. `_update_ui()` stores `self._last_data = data` and updates all widgets

This keeps the Textual event loop responsive — no I/O, subprocesses, or network calls on the main thread.

### Panels

| # | Panel | Refresh | Height | Data source |
|---|-------|---------|--------|-------------|
| 1 | SystemPanel | 2s | 16 | `psutil` (CPU via `interval=1`, mem, disk, load, uptime, SSH sessions, processes) |
| 2 | NetworkPanel | 3s | 12 | `psutil` (interfaces, IO counters, connections) + `ip route` + `/etc/resolv.conf` |
| 3 | GitPanel | 30s | 12 | `os.walk(~/)` to discover repos + `subprocess` git commands per repo |
| 4 | DockerPanel | 5s | 10 | Cached Docker client (container list, ports deduplicated for IPv4/IPv6) |
| 5 | HealthPanel | 15s | 12 | `urllib.request` hitting `/api/health`, `/health`, `/` on containers; parallelized via `ThreadPoolExecutor` |
| 6 | ClaudePanel | 30s | 30 | `~/.claude/stats-cache.json` + Anthropic OAuth API |

All panels use `@work(thread=True)` to fetch data off the main thread.

### Claude usage data

Two data sources feed ClaudePanel:
- **Local cache** (`~/.claude/stats-cache.json`): total sessions/messages, per-model token breakdown, daily activity
- **Live API** (`https://api.anthropic.com/api/oauth/usage`): real-time utilization percentages and reset times for 5h/7d windows. Authenticated via OAuth bearer token from `~/.claude/.credentials.json`. Requires header `anthropic-beta: oauth-2025-04-20`.

API values for `extra_usage` credits are in cents (divide by 100 for dollars).

### Git repo discovery

`get_git_repos()` uses `os.walk()` from `~/` to recursively find all git repos under the user's home directory. It prunes hidden directories and common non-project dirs (`.cache`, `node_modules`, `.venv`, etc.) for performance. Repo names show paths relative to `~` (e.g., `apps/my-project`, `mcps/my-mcp`).

### Docker client caching

`_get_docker_client()` creates a single Docker client on first use and reuses it across all refreshes. Both `get_docker_containers()` and `get_health_checks()` share the same cached client.

### Health check parallelism

`get_health_checks()` collects all container targets first, then checks them concurrently using `ThreadPoolExecutor` (max 10 workers). Each container is checked by `_check_single_container()` which tries `/api/health`, `/health`, `/` in order.

### Helper functions

- `format_bytes()`, `format_tokens()`, `format_duration_ms()` — unit formatting
- `_format_uptime()` — system uptime from `psutil.boot_time()`
- `_count_ssh_sessions()` — counts sshd child processes
- `_time_until(iso_ts, show_days)` — countdown to ISO timestamp; use `show_days=True` for 7d windows
- `_make_bar(pct)` — Rich-markup progress bar, color thresholds: green <70%, yellow 70-90%, red >=90%

### Important notes

- `psutil.cpu_percent(interval=1)` is used (not `interval=0`) because the call runs in a worker thread where the global "last call" state is unreliable. The 1s block is fine since it's off the main thread.
- `_tick_refresh()` always calls `refresh_data()` even when collapsed, so cached summaries stay current.

### Styling

`dashboard.tcss` controls layout. Panels use fixed heights inside a `VerticalScroll` (`#panel-scroll`) so the dashboard scrolls on small terminals. Borders are color-coded: green (system), cyan (network), magenta (git), dodgerblue (docker), yellow (health), orangered (claude). `.collapsed` class sets `height: auto`.

## Key bindings

- `1`-`6` = toggle panel collapsed/expanded
- `Ctrl+Up`/`Ctrl+Down` = reorder focused panel
- `Tab` = cycle focus between panels
- `r` = refresh all expanded panels
- `q` = quit
