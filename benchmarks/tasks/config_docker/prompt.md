# Docker Compose for Python Web App

Create a Docker Compose configuration for a Python web application with PostgreSQL.

## Requirements

- `docker-compose.yml` defining two services: `app` (Python/FastAPI) and `db` (PostgreSQL 16)
- `app` service: build from `./app` directory, expose port 8000, depends on `db`
- `db` service: postgres:16 image, expose port 5432, persistent volume for data
- Environment variables for `app`: DATABASE_URL pointing to the `db` service
- Environment variables for `db`: POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
- Health check on the `db` service
- A `.dockerignore` file excluding `__pycache__`, `.env`, `.git`, `*.pyc`
- Network named `app-network` connecting both services

## Output Files

- `docker-compose.yml`: Docker Compose configuration
- `.dockerignore`: Docker ignore file
