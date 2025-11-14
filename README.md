## **Matrioska v2 - LLM Orchestration System with File-Based Architecture**
![Matrioska](https://live.staticflickr.com/8646/16075618524_6f3b5b199e_b.jpg)

-----

### 👤 Author: **Adriel D. S. Andrade**
- [LinkedIn](https://www.linkedin.com/in/adriel-domingues-de-souza-andrade/)

- [Github repo](https://github.com/adriel007/matrioska)

- [Google Colab](https://colab.research.google.com/drive/1Vq3b7Xu5z2Un0n3_6_dQVYWQrX4fsK0j#scrollTo=3uw0inKtuNS0)

### 📋 Overview

Matrioska v2 is an **advanced orchestration system for large language models (LLMs)** that implements a modular architecture based on files with shared state. Inspired by the concept of Russian nesting dolls, the system decomposes complex tasks into specialized files that communicate via a **shared whiteboard** (`shared_state`).

-----

### 🎯 Key Features

  * **📁 File-Based Architecture:** Automatic decomposition of projects into ordered files.
  * **🧠 Shared State:** Communication system between files via `shared_state`.
  * **💾 Full Persistence:** Checkpoints of architecture and state between executions.
  * **⚡ Sequential Generation:** Each file is generated in dependency order.
  * **🔗 Selective Context:** Files access only relevant information from predecessors.
  * **📦 Optimized Code:** Focus on minimal, complete, and efficient code using CDNs.

-----

### 🏗️ Architecture

#### Core Components

  * **`LocalLLM`** - Wrapper for Mistral models with 4-bit quantization.
  * **`MatrioskaOrchestrator`** - Main pipeline orchestrator.
  * **`ContextManager`** - Manages shared state and persistence.
  * **`Architecture`** - Data structure for file-based planning.
  * **`FileSpec`** - Individual file specification.
  * **`FileArtifact`** - Generated file artifact.

#### Execution Flow

$$
\begin{array}{ccc}
\text{PHASE 1: ARCHITECTURE} & \rightarrow & \text{PHASE 2: CODE GENERATION} \\
\downarrow & & \downarrow \\
\text{File Decomposition} & & \text{Sequential Generation} \\
& & \text{by Order/Dependency}
\end{array}
$$

-----

### 🚀 How to Use

#### Installation

```bash
pip install -q json-repair transformers accelerate bitsandbytes torch sentencepiece protobuf
```

#### Environment Cleanup (Optional)

```bash
!rm -rf /content/log
!rm -rf /content/matrioska_artifacts
!rm -rf /content/matrioska_checkpoints
```

#### Basic Execution

```python
from matrioska_v2 import LocalLLM, MatrioskaOrchestrator

# Initialize model
llm = LocalLLM("mistralai/Mistral-7B-Instruct-v0.3")
orchestrator = MatrioskaOrchestrator(llm, base_path="/content")

# Execute task
result = orchestrator.run("Create a library management system with authentication and dashboard")
```

#### Directory Structure

```
/content/
├── log/                        # Prompt and response logs
│   └── log.txt                # Complete generation history
├── matrioska_artifacts/        # Generated files
│   ├── index.html
│   ├── styles.css
│   └── app.js
└── matrioska_checkpoints/      # State and architecture
    ├── shared_state.json       # Shared whiteboard
    └── architecture.json       # Architectural plan
```

-----

### 📖 File System

#### File Specification (`FileSpec`)

```python
@dataclass
class FileSpec:
    name: str                          # Name without extension
    extension: str                     # File extension
    order: int                         # Creation order (1, 2, 3...)
    shared_state_writes: List[str]     # Info this file defines
    shared_state_reads: List[str]      # Info this file needs
    content: str                       # Code generation prompt
    details: str                       # Functional requirements
```

#### Architecture Example

```json
{
  "instructs": {
    "files": [
      {
        "name": "index",
        "extension": "html",
        "order": 1,
        "shared_state_writes": ["element_ids", "page_structure"],
        "shared_state_reads": [],
        "content": "Generate complete HTML structure for library system...",
        "details": "Responsive layout, login form, book catalog, dashboard"
      },
      {
        "name": "styles",
        "extension": "css",
        "order": 2,
        "shared_state_writes": ["css_classes", "color_scheme"],
        "shared_state_reads": ["element_ids", "page_structure"],
        "content": "Generate complete CSS using Tailwind CDN...",
        "details": "Modern design, dark mode, mobile-first"
      },
      {
        "name": "app",
        "extension": "js",
        "order": 3,
        "shared_state_writes": ["api_endpoints", "storage_keys"],
        "shared_state_reads": ["element_ids", "css_classes"],
        "content": "Generate JavaScript with authentication logic...",
        "details": "JWT auth, localStorage, CRUD operations"
      }
    ]
  }
}
```

#### Shared State Communication Example

```
# File 1 (HTML) generates IDs
SHARED_STATE_UPDATE:
{
  "element_ids": ["#loginForm", "#bookList", "#dashboardStats"],
  "page_structure": {
    "login": "section#login",
    "catalog": "section#catalog",
    "dashboard": "section#dashboard"
  }
}

# File 2 (CSS) automatically consumes IDs
# The ContextManager provides only the keys specified in shared_state_reads
```

-----

### 🔧 Model Configuration

#### 4-bit Quantization

```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

#### Generation Parameters

  * `max_new_tokens`: 20,000 (configurable via `_MAX_TOKEN_`)
  * `temperature`: 0.3
  * `top_p`: 0.85
  * `do_sample`: True
  * `pad_token_id`: Auto (`eos_token_id`)

-----

### 📊 Architecture Prompt

The system uses **`ARCHITECT_SYSTEM_PROMPT`** which instructs the LLM to:

  * Decompose the task into independent files
  * Define creation order based on dependencies
  * Specify contracts via `shared_state_reads`/`writes`
  * Generate complete prompts for each file
  * Focus on minimal code and use of CDNs/libraries

#### Mandatory Prompt Rules

  * Strict JSON structure with `instructs` root
  * `order` field defining creation sequence
  * `shared_state_writes`: information the file defines
  * `shared_state_reads`: information the file needs
  * `content`: complete code generation prompt
  * `details`: functional and non-functional requirements

-----

### 💡 Use Cases

#### Complete Web System

```python
result = orchestrator.run('''
Create a complete e-commerce system with:
- Product catalog with search
- Shopping cart functionality
- User authentication
- Admin dashboard
- Responsive design with Tailwind CDN
''')
```

#### React/Vue Application

```python
result = orchestrator.run('''
Build a task management app using React CDN with:
- Component-based architecture
- State management
- CRUD operations
- LocalStorage persistence
''')
```

#### Data Dashboard

```python
result = orchestrator.run('''
Create an analytics dashboard with:
- Chart.js for visualizations
- Real-time data updates
- Export to CSV functionality
- Responsive grid layout
''')
```

-----

### 🎨 Output Example

```
================================================================================
🪆 MATRIOSKA ORCHESTRATOR - File-Based Architecture
================================================================================

🏗️  PHASE 1: ARCHITECTURE
--------------------------------------------------------------------------------
📋 Task: 'Create a library management system with authentication and dashboard'

✓ Project: Project_3_Files
✓ Files: 3
   1. index.html 📖[] ✍️['element_ids', 'page_structure']
   2. styles.css 📖['element_ids', 'page_structure'] ✍️['css_classes', 'color_scheme']
   3. app.js 📖['element_ids', 'css_classes'] ✍️['api_endpoints', 'storage_keys']

⚡ PHASE 2: CODE GENERATION
--------------------------------------------------------------------------------

🎯 Generating: index.html (Order: 1)
💾 index.html → /content/matrioska_artifacts/index.html
🧠 [SHARED STATE] Updated: ['element_ids', 'page_structure']
   ✍️ Wrote: ['element_ids', 'page_structure']
   ✓ Generated (2847 chars)

🎯 Generating: styles.css (Order: 2)
   📖 Reading context: ['element_ids', 'page_structure']
💾 styles.css → /content/matrioska_artifacts/styles.css
🧠 [SHARED STATE] Updated: ['css_classes', 'color_scheme']
   ✍️ Wrote: ['css_classes', 'color_scheme']
   ✓ Generated (1923 chars)

🎯 Generating: app.js (Order: 3)
   📖 Reading context: ['element_ids', 'css_classes']
💾 app.js → /content/matrioska_artifacts/app.js
🧠 [SHARED STATE] Updated: ['api_endpoints', 'storage_keys']
   ✍️ Wrote: ['api_endpoints', 'storage_keys']
   ✓ Generated (3456 chars)

✅ FINAL RESULT
================================================================================

📦 Project_3_Files

📂 Generated Files: 3
   1. index.html
   2. styles.css
   3. app.js

🧠 SharedState Keys: ['element_ids', 'page_structure', 'css_classes', 'color_scheme', 'api_endpoints', 'storage_keys']
================================================================================

📁 Artifacts: /content/matrioska_artifacts
🧠 SharedState: /content/matrioska_checkpoints/shared_state.json
```

-----

### 🔄 State Management

#### Shared State

  * **Persistent:** Saved in `shared_state.json` between executions.
  * **Structured:** JSON-serializable dictionary.
  * **Selective:** Files access only keys specified in `shared_state_reads`.
  * **Incremental:** Updated during the generation of each file.

#### Checkpoints

  * **Architecture:** `architecture.json` - Complete project plan
  * **SharedState:** `shared_state.json` - Current shared state
  * **Artifacts:** Individual files in `matrioska_artifacts/`
  * **Logs:** Complete history of prompts and responses in `log/log.txt`

#### Shared State Example (`shared_state.json`)

```json
{
  "element_ids": ["#loginForm", "#bookList", "#dashboard"],
  "page_structure": {
    "login": "section#login",
    "catalog": "section#catalog"
  },
  "css_classes": ["btn-primary", "card", "nav-item"],
  "color_scheme": {
    "primary": "#3b82f6",
    "secondary": "#8b5cf6"
  },
  "api_endpoints": {
    "login": "/api/auth/login",
    "books": "/api/books"
  },
  "storage_keys": ["authToken", "currentUser"]
}
```

#### 📦 SharedState Updates Extraction

The system automatically detects updates in the format:

```
// At the end of the generated code
SHARED_STATE_UPDATE:
{
  "key1": "value1",
  "key2": ["item1", "item2"]
}
```

This marker is:

  * Extracted and processed by the `ContextManager`
  * Removed from the final code
  * Persisted in `shared_state.json`

#### 📄 Returned API

```python
result = orchestrator.run("Create app...")

# Returns a dictionary with:
{
  "architecture": Architecture,     # Object with the project plan
  "artifacts": List[FileArtifact], # List of generated files
  "shared_state": Dict[str, Any]   # Final shared state
}
```

-----

### 🛠️ Technical Requirements

  * **GPU:** NVIDIA T4 (8GB VRAM) or superior
  * **RAM:** 12GB+ recommended
  * **Python:** 3.8+
  * **Libraries:**
      * `transformers` (Hugging Face)
      * `torch` (PyTorch)
      * `bitsandbytes` (Quantization)
      * `accelerate` (Optimization)
      * `json-repair` (Robust Parsing)
      * `sentencepiece`, `protobuf` (Tokenization)

-----

### 🔍 Logging and Debug

All prompts and responses are saved in `/content/log/log.txt`:

```
PROMPT:
==========================================
[Complete prompt sent to LLM]
==========================================
RESULT:
==========================================
[LLM Response]
```

-----

### 🎯 Best Practices

  * **File Order:** HTML/DB first → CSS/Styles → JS/Logic → API/Backend
  * **SharedState:** Define clear contracts between files (IDs, classes, routes)
  * **Detailed Prompts:** The `content` field must be a complete generation prompt
  * **CDNs:** Prioritize libraries via CDN to reduce complexity
  * **Minimal Code:** Focus on minimal and functional code

-----

### 🔮 Differences from v1

| Aspect | v1 (Modules) | v2 (Files) |
| :--- | :--- | :--- |
| **Basic Unit** | `ModuleSpec` | `FileSpec` |
| **Final Integration** | Artifact assembly | Independent files |
| **Structure** | 3 phases | 2 phases |
| **Focus** | Conceptual modularity | Practical code generation |
| **Output** | Integrated result | Separate files |

-----

### 📄 License

This project is intended for research and educational development purposes.

**Matrioska v2: Transforming ideas into structured code 🪆✨**