# REST API with JWT Authentication

Create a FastAPI application with JWT-based authentication for protected endpoints.

## Files

- `main.py`: FastAPI app entry point. Includes CORS, registers routers from `routes.py`.

- `models.py`: Pydantic models:
  - `UserCreate(username: str, password: str)`
  - `UserResponse(id: int, username: str)`
  - `TokenResponse(access_token: str, token_type: str = "bearer")`
  - In-memory user store: `users_db: dict[str, dict]` (username → hashed password + id)

- `auth.py`: Authentication utilities:
  - `SECRET_KEY = "benchmark-secret-key-123"` (hardcoded for testing)
  - `ALGORITHM = "HS256"`
  - `def create_token(username: str) -> str` — creates JWT with `sub` claim
  - `def verify_token(token: str) -> str | None` — verifies and returns username or None
  - Use PyJWT library (`import jwt`)

- `middleware.py`: FastAPI dependency:
  - `def get_current_user(token: str = Depends(oauth2_scheme)) -> dict` — extracts bearer token, verifies, returns user info
  - `oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")`
  - Raises HTTPException 401 if invalid token

- `routes.py`: API endpoints:
  - `POST /auth/register` — create user, return TokenResponse
  - `POST /auth/login` — verify credentials, return TokenResponse
  - `GET /users/me` — protected endpoint, returns current user info
  - `GET /health` — public, returns `{"status": "ok"}`

- `requirements.txt`: fastapi, uvicorn, pyjwt, passlib (or bcrypt)

## Key cross-file dependencies

- `routes.py` imports `models` (UserCreate, UserResponse, TokenResponse, users_db), `auth` (create_token, verify_token), `middleware` (get_current_user)
- `main.py` imports `routes` (router)
- `middleware.py` imports `auth` (verify_token), `models` (users_db)
- `auth.py` is standalone (imported by others)
