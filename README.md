<div align="center">
  <h1 align="center">🗺️ repo-map</h1>
  <p align="center">
    <a href="https://pypi.org/project/repo-map/">
      <img src="https://img.shields.io/pypi/v/repo-map.svg" alt="PyPI version">
    </a>
    <a href="https://pypi.org/project/repo-map/">
      <img src="https://img.shields.io/pypi/pyversions/repo-map.svg" alt="Python versions">
    </a>
    <a href="https://pypi.org/project/repo-map/">
      <img src="https://img.shields.io/pypi/l/repo-map.svg" alt="PyPI license">
    </a>
    <a href="https://pypi.org/project/repo-map/">
      <img src="https://img.shields.io/pypi/dm/repo-map.svg" alt="PyPI downloads">
    </a>
  </p>
  <p align="center">
    An intelligent repository mapper that uses AI to create comprehensive, visual summaries of your codebase.
  </p>
</div>

<hr>

**repo-map** is an advanced tool for generating comprehensive, AI-enhanced summaries of software repositories. It provides developers with valuable insights into project structures, file purposes, and potential considerations across various programming languages. Using efficient caching, repo-map only processes files that have changed since the last run, making it ideal for continuous use in evolving projects. This tool not only aids in understanding and documenting codebases but can also assist LLM agents in writing accurate and functional code within your existing project structure.

## 🌟 Key Features

-   **📊 Detailed Repository Summaries:** Generates a tree-like visualization of your repository structure.
-   **🧠 AI-Powered Enhancements:** Get AI-generated descriptions, developer considerations, and more.
-   **🌐 Multi-Language Support:** Analyzes code structure across Python, Java, JavaScript, TypeScript, and more.
-   **🚀 Efficient Caching:** Uses SQLite to only process files that have changed since the last run.
-   **📝 Markdown Output:** Generates a clean Markdown file for easy sharing and documentation.
-   **🔒 Gitignore Respect:** Respects your root `.gitignore` file and includes a robust set of default ignore patterns.
-   **⚡ Async Processing:** Utilizes asynchronous processing for improved performance.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/cyanheads/repo-map.git
    cd repo-map
    ```

2.  **Install dependencies using Poetry:**
    ```bash
    poetry install
    ```

## 🚀 Usage

To generate a repository map, run the following command from the project root:

```bash
poetry run repo-map <repository_path> [options]
```

Replace `<repository_path>` with the path to the repository you want to analyze.

### Options

-   `-y`, `--yes`: Automatically accept the disclaimer and proceed without prompting.
-   `--model MODEL`: Specify the OpenRouter LLM model to use (default: `google/gemini-2.5-flash-preview-09-2025`).
-   `--concurrency INT`: Set the number of concurrent API calls (default: 3).

### Examples

```bash
# Basic usage
repo-map /path/to/your/repo

# Use a specific model
repo-map /path/to/your/repo --model "google/gemini-2.5-flash-preview-09-2025"

# Auto-accept disclaimer
repo-map /path/to/your/repo -y
```

## 🐍 Example: Snake Game Repository Map

Here's an example of a repo-map generated for an advanced Snake game implemented in Python:

```markdown
/ (SSSnakeGame)
├── main.py (Python)
│   ├── Description: Entry point for the Snake game, initializes the game environment and runs the main event loop.
│   ├── Developer Consideration: "The game loop is tightly bound to Pygame's event system; any significant changes will require familiarity with Pygame's architecture."
│   ├── Maintenance Flag: Stable
│   ├── Architectural Role: Entrypoint
│   ├── Code Quality Score: 8/10
│   ├── Refactoring Suggestions: "Isolate game state management from the rendering logic to improve testability and reduce complexity."
│   └── Security Assessment: "None"
├── config.py (Python)
│   ├── Description: Centralizes all static configuration parameters for the game, such as screen dimensions, colors, and snake speed.
│   ├── Developer Consideration: "Changing screen dimensions may require adjustments to the food spawning logic to ensure it appears within bounds."
│   ├── Maintenance Flag: Volatile
│   ├── Architectural Role: Configuration
│   ├── Code Quality Score: 9/10
│   ├── Refactoring Suggestions: "Consider using a more structured configuration format like YAML or JSON for easier management, especially if settings become more complex."
│   └── Security Assessment: "None"
├── assets/
│   ├── images/
│   │   ├── snake_head.png (Image)
│   │   └── food.png (Image)
│   └── sounds/
│       ├── eat.wav (Audio)
│       └── game_over.mp3 (Audio)
├── requirements.txt (Text)
│   └── Description: Lists all Python package dependencies required to run the project, such as `pygame`.
└── README.md (Markdown)
    └── Description: Provides a comprehensive overview of the project, including setup instructions, gameplay details, and contribution guidelines.
└──────────────
```

## 🔐 Configuration

Before using repo-map, you need to set up your OpenRouter API key. Set the following environment variable:

```bash
export OPENROUTER_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual OpenRouter API key.

## 🧩 How It Works

1.  **Walks** through the repository directory structure.
2.  **Analyzes** file contents and extracts key information (imports, functions, classes).
3.  **Utilizes** an LLM (via OpenRouter) to generate descriptions and developer considerations.
4.  **Caches** results in SQLite for efficient processing of unchanged files.
5.  **Generates** a comprehensive tree-like structure of the repository.
6.  **Saves** the output as a Markdown file for easy viewing and sharing.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/cyanheads/repo-map/issues).

## ️ License

This project is licensed under the Apache 2.0 License. See the `LICENSE` file in the root directory of this project for the full license text.
