# clod-compu-stats

Terminal dashboard that shows system metrics, Docker container status, and Claude usage stats in a single view. Built with Python + [Textual](https://github.com/Textualize/textual).

## Panels

- **System Monitor** — CPU, memory, disk usage + top 8 processes by CPU
- **Docker Containers** — status, name, image, ports, uptime for all containers
- **Claude Usage** — live utilization bars (5h/7d windows) from the Anthropic API, plus token breakdowns and daily activity from `~/.claude/stats-cache.json`

## Setup

```bash
# Requires python3.12-venv: sudo apt install python3.12-venv
./setup.sh
```

## Usage

```bash
./run.sh
```

### Keybindings

| Key | Action      |
|-----|-------------|
| `r` | Refresh all |
| `q` | Quit        |

## Requirements

- Python 3.12+
- Docker (optional — panel shows graceful fallback)
- Claude Code credentials at `~/.claude/.credentials.json` for live usage data
