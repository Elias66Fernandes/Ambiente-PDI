import os

# Configurações do Gunicorn para produção
workers = int(os.environ.get('GUNICORN_WORKERS', 1))
threads = int(os.environ.get('GUNICORN_THREADS', 4))
timeout = int(os.environ.get('GUNICORN_TIMEOUT', 120))
bind = f"0.0.0.0:{os.environ.get('PORT', 5000)}"

# Configurações de logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Configurações de worker
worker_class = 'gthread'
keepalive = 65
