# Gunicorn configuration file for WebSocket support
import multiprocessing

# Worker settings
worker_class = "eventlet"
workers = multiprocessing.cpu_count()
worker_connections = 1000
timeout = 300

# Logging
loglevel = "debug"
accesslog = "-"
errorlog = "-"
capture_output = True

# Server mechanics
daemon = False
reload = True
bind = "0.0.0.0:5000"

# SSL (if needed)
keyfile = None
certfile = None

def on_starting(server):
    """Log when the server starts"""
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.info("Starting Gunicorn with eventlet worker class")
