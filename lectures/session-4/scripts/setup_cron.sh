#!/bin/bash
#
# setup_cron.sh — Install crontab entries for the intraday production runner.
#
# Usage:
#   cd lectures/session-4/scripts
#   chmod +x setup_cron.sh
#   ./setup_cron.sh
#
# This installs the intraday cadence (weekdays only, ET timezone):
#
#   Engine fires (decide + auto-execute small trades / queue exceptions):
#     09:30 -- open fire (--mode=engine)
#     10:00, 10:30, 11:00, 11:30, 12:00, 12:30,
#     13:00, 13:30, 14:00, 14:30, 15:00, 15:30  -- intraday fires (--mode=engine)
#     16:00 -- close fire (--mode=engine_close); writes EOD tape and tomorrow's ticket
#
#   News scoring fires (hourly during the session):
#     10:00, 11:00, 12:00, 13:00, 14:00, 15:00, 16:00  (--mode=hourly)
#
#   Next-day execution of class-signed ticket:
#     09:35 -- (--mode=execute_signed_ticket); submits the signed ticket
#              from the prior session's TomorrowsTicket sign-off.
#
# Total fires per trading day: 14 engine + 7 news + 1 next-day = 22.
#
# To remove: run `crontab -e` and delete lines marked with [AI-FINANCE], or
# run `crontab -l | grep -v '\[AI-FINANCE\]' | crontab -`.
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SESSION_DIR="$(dirname "$SCRIPT_DIR")"
JULIA="$(which julia)"
ENV_FILE="$HOME/.ecornell-s4-env"

if [ -z "$JULIA" ]; then
    echo "Error: julia not found in PATH."
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found. See cron_setup_instructions.md."
    exit 1
fi

LOG="$SESSION_DIR/data/production-log.txt"

echo "Script directory: $SCRIPT_DIR"
echo "Session directory: $SESSION_DIR"
echo "Julia binary: $JULIA"
echo "Env file: $ENV_FILE"
echo "Log file: $LOG"
echo ""
echo "Adding cron entries..."

# Each cron entry runs under /bin/bash so it can source the env file.
# The env file exports ANTHROPIC_API_KEY (and any other secrets cron needs).
PREFIX="/bin/bash -c 'source $ENV_FILE && cd $SCRIPT_DIR &&"
SUFFIX="'"
ENG_CMD="$PREFIX $JULIA --project=$SESSION_DIR production_runner.jl --mode=engine >> $LOG 2>&1$SUFFIX"
CLOSE_CMD="$PREFIX $JULIA --project=$SESSION_DIR production_runner.jl --mode=engine_close >> $LOG 2>&1$SUFFIX"
EXEC_CMD="$PREFIX $JULIA --project=$SESSION_DIR production_runner.jl --mode=execute_signed_ticket >> $LOG 2>&1$SUFFIX"
NEWS_CMD="$PREFIX $JULIA --project=$SESSION_DIR news_scorer.jl --mode=hourly >> $LOG 2>&1$SUFFIX"

# Append to existing crontab (preserving non-AI-FINANCE entries) -
(crontab -l 2>/dev/null | grep -v "\[AI-FINANCE\]"; cat <<EOF
# === Engine fires ===
30 9     * * 1-5 $ENG_CMD    # [AI-FINANCE] engine-open
0,30 10-15 * * 1-5 $ENG_CMD  # [AI-FINANCE] engine-intraday
0  16    * * 1-5 $CLOSE_CMD  # [AI-FINANCE] engine-close
# === News scoring (hourly) ===
0  10-16 * * 1-5 $NEWS_CMD   # [AI-FINANCE] news-hourly
# === Next-day execution of class-signed ticket ===
35 9     * * 1-5 $EXEC_CMD   # [AI-FINANCE] execute-signed-ticket
EOF
) | crontab -

echo "Cron entries installed. Current AI-FINANCE crontab:"
echo "---"
crontab -l | grep "\[AI-FINANCE\]"
echo "---"
echo ""
echo "To remove all AI-FINANCE entries:"
echo "  crontab -l | grep -v '\[AI-FINANCE\]' | crontab -"
