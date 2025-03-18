from app import app, socketio
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting server with eventlet WebSocket support")
    # ALWAYS serve the app on port 5000
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=True, log_output=True)