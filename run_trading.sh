#!/bin/bash
cd /Arbos
# -u: unbuffered stdout/stderr for PM2 logs
exec /usr/bin/python3 -u -m trading.live --interval 3600 --symbols BTC,ETH,SOL
