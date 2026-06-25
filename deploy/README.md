# Deployment

## Quick start (any Linux with systemd)

```bash
# Install deps
pip install -r requirements.txt

# Install and start the systemd service
sudo cp deploy/object-detection.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now object-detection
sudo systemctl status object-detection
```

The service auto-starts on boot and restarts on crash.

## Without systemd (containers / Docker)

```bash
pip install supervisor
echo_supervisord_conf > /etc/supervisor/supervisord.conf
echo -e '\n[include]\nfiles = /etc/supervisor/conf.d/*.conf' >> /etc/supervisor/supervisord.conf
mkdir -p /etc/supervisor/conf.d
cp deploy/supervisor-object-detection.conf /etc/supervisor/conf.d/object-detection.conf
supervisord -c /etc/supervisor/supervisord.conf
supervisorctl -c /etc/supervisor/supervisord.conf status
```

## Logs

| Method | Log location |
|--------|-------------|
| systemd | `journalctl -u object-detection -f` |
| supervisord | `/var/log/object-detection.log` |

## Access

Open **http://localhost:8000** (or replace `localhost` with the machine's IP for LAN access).
