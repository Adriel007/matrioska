# Full-Stack Todo Application

Build a complete todo app with Python FastAPI backend, SQLite database, and vanilla JavaScript frontend.

## Backend (Python/FastAPI)

- `main.py`: FastAPI app entry point, CORS enabled, serves static files from `./static`
- `models.py`: Pydantic model `TodoCreate(title: str)` and `TodoResponse(id: int, title: str, done: bool, created_at: str)`. Also SQLAlchemy model `Todo` with columns: id (Integer PK), title (String), done (Boolean default False), created_at (String ISO format)
- `database.py`: SQLite database setup using SQLAlchemy. Creates tables. Provides `get_db()` dependency. Connection string: `sqlite:///./todos.db`
- `routes.py`: REST endpoints:
  - `GET /api/todos` → list all todos
  - `GET /api/todos/{id}` → get one todo
  - `POST /api/todos` → create todo
  - `PUT /api/todos/{id}` → toggle done status
  - `DELETE /api/todos/{id}` → delete todo

## Frontend (vanilla JS + HTML)

- `static/index.html`: Single page app with:
  - Input field + Add button to create todos
  - Todo list with checkboxes to toggle done, delete buttons
  - Clean, usable UI with basic CSS styling inline or in `<style>` tag
- `static/app.js`: JavaScript that:
  - Fetches todos from `GET /api/todos` on page load
  - Handles add, toggle, delete operations via fetch() to the API
  - Updates the DOM without page reload
  - Handles errors with alert() or console.error

## Output Files

- `main.py`
- `models.py`
- `database.py`
- `routes.py`
- `static/index.html`
- `static/app.js`
- `requirements.txt` (fastapi, uvicorn, sqlalchemy)
