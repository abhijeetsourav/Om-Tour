#!/bin/bash
set -e

echo "Configuring git..."
git config --global core.autocrlf input

echo "Installing pnpm..."
corepack enable
corepack prepare pnpm@latest --activate

echo "Installing Poetry..."
curl -sSL https://install.python-poetry.org | python3 -

export PATH="$HOME/.local/bin:$PATH"

echo "Creating env files..."

# Backend env
mkdir -p agent
touch agent/.env

cat <<EOF > agent/.env
OPENAI_API_KEY=
GROQ_API_KEY=
SERPAPI_API_KEY=
EOF

# Frontend env
mkdir -p ui
touch ui/.env.local

cat <<EOF > ui/.env.local
NEXT_PUBLIC_COPILOTKIT_API_URL=
NEXT_PUBLIC_MAPBOX_TOKEN=
EOF

echo "Installing frontend dependencies..."
cd ui
pnpm install

echo "Installing backend dependencies..."
cd ../agent
poetry install || pip install -e .

echo "Setup complete."