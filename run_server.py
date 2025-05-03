from waitress import serve
from app import app

if __name__ == '__main__':
    # Serve on all interfaces on port 5000
    serve(app, host='0.0.0.0', port=5000, threads=6)
    print("Waitress server started on http://localhost:5000")