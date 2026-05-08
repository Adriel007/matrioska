# REST API for Book Inventory

Create a FastAPI REST API for managing a book inventory. Use in-memory storage (Python dict/list).

## Requirements

- `GET /books` — list all books
- `GET /books/{id}` — get a single book by ID
- `POST /books` — create a new book (JSON body: title, author, year)
- `PUT /books/{id}` — update a book
- `DELETE /books/{id}` — delete a book
- Each book has: id (auto-incremented), title (str), author (str), year (int)
- Return proper HTTP status codes (200, 201, 404, 422)
- Include input validation (title required, year must be positive integer)

## Output Files

- `main.py`: FastAPI application
- `requirements.txt`: Dependencies (fastapi, uvicorn)
