module.exports = {
  apps: [
    {
      name: 'arbos',
      script: '/Arbos/.venv/bin/python3',
      args: 'arbos.py',
      cwd: '/Arbos',
      interpreter: 'none',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      env: {
        NODE_ENV: 'production'
      },
      error_file: '/Arbos/logs/arbos-error.log',
      out_file: '/Arbos/logs/arbos-out.log',
      log_file: '/Arbos/logs/arbos.log',
      time: true
    },
    {
      name: 'trading',
      script: '/Arbos/run_trading.sh',
      interpreter: 'none',
      cwd: '/Arbos',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      min_uptime: '30s',
      restart_delay: 5000,
      env: {
        PYTHONPATH: '/Arbos'
      },
      error_file: '/Arbos/logs/trading-error.log',
      out_file: '/Arbos/logs/trading-out.log',
      log_file: '/Arbos/logs/trading.log',
      time: true
    }
  ]
};
