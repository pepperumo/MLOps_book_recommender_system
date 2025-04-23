#!/bin/bash
set -e  # Exit immediately on error

echo "🚀 Starting services for Hugging Face deployment..."

# Ensure necessary directories exist
mkdir -p /app/logs /app/models /app/data/processed /app/data/raw /app/data/features

# Check nginx configuration
echo "🔍 Checking nginx configuration..."
nginx -t

# Check if model exists, if not, provide instructions
if [ ! -f "$MODEL_PATH" ]; then
    echo "⚠️ Model file not found at $MODEL_PATH"
    echo "📝 You need to upload your model file to the Hugging Face Space:"
    echo "1. Go to your Space's 'Files' tab"
    echo "2. Upload your collaborative.pkl file to the /app/models/ directory"
    echo "3. Restart your Space"
    
    # Create a basic HTML file to display instructions when accessed through the browser
    cat > /usr/share/nginx/html/index.html <<EOL
<!DOCTYPE html>
<html>
<head>
    <title>Book Recommender System - Setup Required</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; line-height: 1.6; }
        .container { padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        h1 { color: #2c3e50; }
        .steps { background-color: #f8f9fa; padding: 15px; border-radius: 5px; }
        .warning { color: #e74c3c; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Book Recommender System - Setup Required</h1>
        <p class="warning">⚠️ The model file is missing from this deployment.</p>
        <p>To complete the setup of this application, you need to:</p>
        <div class="steps">
            <h3>Upload the necessary files:</h3>
            <ol>
                <li>Go to your Space's 'Files' tab</li>
                <li>Upload your collaborative.pkl file to the /app/models/ directory</li>
                <li>If needed, upload your data files to the /app/data/ directory</li>
                <li>Restart your Space</li>
            </ol>
        </div>
        <p>Once these steps are completed, refresh this page to see your application.</p>
    </div>
</body>
</html>
EOL
fi

# Check for frontend build files
echo "🔍 Checking frontend files..."
ls -la /usr/share/nginx/html

# Create compatibility link for imports
mkdir -p /app/src/api
echo "from src.fastAPI.api import app" > /app/src/api/__init__.py

# Set up Python environment
export PYTHONPATH="${PYTHONPATH}:/app"
pip install -e .

# Start the FastAPI server in the background
echo "🌐 Starting FastAPI server on port ${PORT:-8000}..."
uvicorn src.fastAPI.api:app --host 0.0.0.0 --port ${PORT:-8000} &
FASTAPI_PID=$!

# Wait a moment for FastAPI to start
sleep 2

# Start nginx in the foreground so we can see any errors
echo "🌐 Starting Nginx server..."
nginx -g 'daemon off;' &
NGINX_PID=$!

# Create a simple proxy that listens on port 7860 and forwards to nginx on port 80
# This is necessary because Hugging Face Spaces expects the app to listen on port 7860
echo "🔄 Setting up proxy from port 7860 to internal services..."
python -c "
import http.server
import socketserver
import urllib.request

PORT = 7860

class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Log the request
        print(f'Received request for: {self.path}')
        
        # Forward requests to nginx running on port 80
        url = f'http://localhost:80{self.path}'
        try:
            response = urllib.request.urlopen(url)
            self.send_response(response.status)
            for header, value in response.getheaders():
                self.send_header(header, value)
            self.end_headers()
            self.wfile.write(response.read())
            print(f'Successfully forwarded request to {url}')
        except Exception as e:
            print(f'Error forwarding request: {e}')
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def do_POST(self):
        # Same for POST requests
        print(f'Received POST request for: {self.path}')
        
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        url = f'http://localhost:80{self.path}'
        
        req = urllib.request.Request(url, data=post_data, method='POST')
        for header in self.headers:
            if header.lower() not in ['content-length', 'host']:
                req.add_header(header, self.headers[header])
        
        try:
            response = urllib.request.urlopen(req)
            self.send_response(response.status)
            for header, value in response.getheaders():
                self.send_header(header, value)
            self.end_headers()
            self.wfile.write(response.read())
            print(f'Successfully forwarded POST request to {url}')
        except Exception as e:
            print(f'Error forwarding POST request: {e}')
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())

with socketserver.TCPServer(('', PORT), ProxyHandler) as httpd:
    print(f'✅ Proxy server running at port {PORT}')
    httpd.serve_forever()
"