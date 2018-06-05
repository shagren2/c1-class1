import os

PORT = os.getenv('PORT', default=5000)
SERVING_WORKERS = os.getenv('SERVING_WORKERS', default=2)
TIMEOUT = os.getenv('TIMEOUT', default=360)

bind = f'0.0.0.0:{PORT}'
workers = SERVING_WORKERS
timeout = TIMEOUT

errorlog = '-'
loglevel = 'debug'
accesslog = '-'

pythonpath = os.getenv('PYTHONPATH')
