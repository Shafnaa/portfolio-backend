export CORS_ALLOW_ORIGIN="http://localhost:5173;http://localhost:8080;http://localhost:3000"
PORT="${PORT:-8080}"

ROOT_DIR="$(dirname "$0")/.."

echo "→ Building frontend..."
(cd "$ROOT_DIR" && bun run build)

echo "→ Starting backend..."
uvicorn portfolio.main:app --port $PORT --host 0.0.0.0 --forwarded-allow-ips '*' --reload
