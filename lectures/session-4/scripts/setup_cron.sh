#!/bin/bash
#
# setup_cron.sh — Install crontab entries for the production runner.
#
# Usage:
#   cd lectures/session-4/scripts
#   chmod +x setup_cron.sh
#   ./setup_cron.sh
#
# This adds 4 cron entries (weekdays only, ET timezone):
#   9:35  — production mode (full pipeline + trade)
#   12:00 — monitor mode (safety check)
#   14:00 — monitor mode (safety check)
#   15:50 — monitor mode (end-of-day check)
#
# To remove: run `crontab -e` and delete the lines marked with [AI-FINANCE].
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JULIA="$(which julia)"

if [ -z "$JULIA" ]; then
    echo "Error: julia not found in PATH."
    exit 1
fi

echo "Script directory: $SCRIPT_DIR"
echo "Julia binary: $JULIA"
echo ""
echo "Adding cron entries..."

# Build the cron lines
PROD_CMD="cd $SCRIPT_DIR && $JULIA --project=../.. production_runner.jl --mode=production >> ../data/production-log.txt 2>&1"
MON_CMD="cd $SCRIPT_DIR && $JULIA --project=../.. production_runner.jl --mode=monitor >> ../data/production-log.txt 2>&1"

# Append to existing crontab (preserving other entries)
(crontab -l 2>/dev/null | grep -v "\[AI-FINANCE\]"; cat <<EOF
35 9  * * 1-5 $PROD_CMD  # [AI-FINANCE] production
0  12 * * 1-5 $MON_CMD   # [AI-FINANCE] monitor-midday
0  14 * * 1-5 $MON_CMD   # [AI-FINANCE] monitor-afternoon
50 15 * * 1-5 $MON_CMD   # [AI-FINANCE] monitor-close
EOF
) | crontab -

echo "Cron entries installed. Current crontab:"
echo "---"
crontab -l | grep "\[AI-FINANCE\]"
echo "---"
echo ""
echo "To remove all AI-FINANCE entries:"
echo "  crontab -l | grep -v '\[AI-FINANCE\]' | crontab -"
