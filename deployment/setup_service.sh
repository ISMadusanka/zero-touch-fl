#!/bin/bash
# =============================================================
# Zero-Touch FL — Server Deployment Script
# =============================================================
# Run this script on your Ubuntu server to set up the FL
# training as a systemd service.
#
# Usage:
#   chmod +x deployment/setup_service.sh
#   sudo ./deployment/setup_service.sh
# =============================================================

set -e

# ---- Configuration (update these to match your server) ----
PROJECT_DIR="/home/ubuntu/zero-touch-fl"
VENV_DIR="${PROJECT_DIR}/venv"
SERVICE_NAME="zero-touch-fl"
SERVICE_USER="ubuntu"
# -----------------------------------------------------------

echo "=================================================="
echo " Zero-Touch FL — Service Setup"
echo "=================================================="

# 1. Check we're running as root
if [ "$EUID" -ne 0 ]; then
  echo "ERROR: Please run with sudo"
  exit 1
fi

# 2. Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
  echo "[1/5] Creating Python virtual environment..."
  sudo -u "$SERVICE_USER" python3 -m venv "$VENV_DIR"
else
  echo "[1/5] Virtual environment already exists."
fi

# 3. Install dependencies
echo "[2/5] Installing Python dependencies..."
sudo -u "$SERVICE_USER" "$VENV_DIR/bin/pip" install --upgrade pip
sudo -u "$SERVICE_USER" "$VENV_DIR/bin/pip" install -r "${PROJECT_DIR}/requirements.txt"

# 4. Copy service file
echo "[3/5] Installing systemd service..."
cp "${PROJECT_DIR}/deployment/zero-touch-fl.service" /etc/systemd/system/${SERVICE_NAME}.service

# Update paths in the service file to match configuration
sed -i "s|WorkingDirectory=.*|WorkingDirectory=${PROJECT_DIR}|" /etc/systemd/system/${SERVICE_NAME}.service
sed -i "s|ExecStart=.*|ExecStart=${VENV_DIR}/bin/python main.py --fresh|" /etc/systemd/system/${SERVICE_NAME}.service
sed -i "s|User=.*|User=${SERVICE_USER}|" /etc/systemd/system/${SERVICE_NAME}.service
sed -i "s|Group=.*|Group=${SERVICE_USER}|" /etc/systemd/system/${SERVICE_NAME}.service

# 5. Reload systemd and enable
echo "[4/5] Reloading systemd..."
systemctl daemon-reload

echo "[5/5] Setup complete!"
echo ""
echo "=================================================="
echo " Quick Reference Commands"
echo "=================================================="
echo ""
echo "  Start training:"
echo "    sudo systemctl start ${SERVICE_NAME}"
echo ""
echo "  Check status:"
echo "    sudo systemctl status ${SERVICE_NAME}"
echo ""
echo "  View live logs:"
echo "    sudo journalctl -u ${SERVICE_NAME} -f"
echo ""
echo "  View last 100 log lines:"
echo "    sudo journalctl -u ${SERVICE_NAME} -n 100"
echo ""
echo "  Stop training:"
echo "    sudo systemctl stop ${SERVICE_NAME}"
echo ""
echo "  Restart training:"
echo "    sudo systemctl restart ${SERVICE_NAME}"
echo ""
echo "  Enable auto-start on boot (optional):"
echo "    sudo systemctl enable ${SERVICE_NAME}"
echo ""
echo "=================================================="
