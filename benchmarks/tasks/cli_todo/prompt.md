# CLI Todo App

Create a Python command-line todo application that stores tasks in a JSON file.

## Requirements

- Add a task: `python todo.py add "Buy groceries"`
- List all tasks: `python todo.py list`
- Mark a task as done: `python todo.py done <task_id>`
- Delete a task: `python todo.py delete <task_id>`
- Tasks stored in a `tasks.json` file in the same directory
- Each task has: id (auto-incremented integer), title (string), done (boolean), created_at (ISO format string)
- Print clear messages for each operation
- Handle missing file gracefully (create it on first use)

## Output Files

- `todo.py`: Main CLI script
- `requirements.txt`: Dependencies (if any beyond stdlib)
