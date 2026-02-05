import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from app import create_app
from flask_cors import CORS



app = create_app()

CORS(app)

if __name__ == "__main__":
    app.run(
	host="0.0.0.0",
        port=5000,
        debug=True
)
