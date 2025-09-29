# 🗺️ repo-map
repo-map is an advanced tool for generating comprehensive, AI-enhanced summaries of software repositories. It provides developers with valuable insights into project structures, file purposes, and potential considerations across various programming languages. Using efficient caching, repo-map only processes files that have changed since the last run, making it ideal for continuous use in evolving projects. This tool not only aids in understanding and documenting codebases but can also assist LLM agents in writing accurate and functional code within your existing project structure.

## 🌟 Features
- 📊 Generates detailed repository structure summaries
- 🧠 AI-powered enhancements:
  - 💡 Developer considerations for potential issues or unique aspects
  - 🗣️ Concise explanations of file purposes and functionality
  - 🔍 Insights into code structure and organization
- 🌐 Analyzes code structure across multiple programming languages
- 🚀 Supports various file types including Python, Java, JavaScript, TypeScript, and more
- 💾 Caching mechanism using SQLite for efficient processing of unchanged files
- 🌳 Tree-like visualization of the repository structure
- 📝 Markdown output for easy sharing and documentation
- 🔒 Respects the root `.gitignore` file and includes a robust set of default ignore patterns
- 🚦 Implements rate limiting and exponential backoff for LLM API calls
- ⚡ Asynchronous processing for improved performance

## 🛠️ Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/cyanheads/repo-map.git
    cd repo-map
    ```
2.  Install dependencies using Poetry:
    ```bash
    poetry install
    ```

## 🚀 Usage
To generate a repository map, run the following command from the project root:
```bash
poetry run repo-map <repository_path> [options]
```
Replace `<repository_path>` with the path to the repository you want to analyze.

### Options:
- `-y,` `--yes`: Automatically accept the disclaimer and proceed without prompting.
- `--model MODEL`: Specify the OpenRouter LLM model to use (default: `google/gemini-2.5-flash-preview-09-2025`).
- `--concurrency INT`: Set the number of concurrent API calls (default: 3).

Examples:
```bash
# Basic usage
repo-map /path/to/your/repo

# Use a specific model
repo-map /path/to/your/repo --model anthropic/claude-3-opus

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

This example demonstrates how repo-map provides a comprehensive overview of a Snake game project, including file descriptions, developer considerations, and key structural information.

## 🔧 Requirements
- Python 3.12+
- [Poetry](https://python-poetry.org/) for dependency management.
- Dependencies are listed in `pyproject.toml`.

## 🔐 Configuration
Before using repo-map, you need to set up your OpenRouter API key. Set the following environment variable:
```bash
export OPENROUTER_API_KEY=your_api_key_here
```
Replace `your_api_key_here` with your actual OpenRouter API key.

## 🧩 How It Works
1. 📂 Walks through the repository directory structure
2. 📝 Analyzes file contents and extracts key information (imports, functions, classes)
3. 🤖 Utilizes an LLM (via OpenRouter) to generate descriptions and developer considerations
4. 🗃️ Caches results in SQLite for efficient processing of unchanged files
5. 📊 Generates a comprehensive tree-like structure of the repository
6. 💾 Saves the output as a Markdown file for easy viewing and sharing

## 🔑 Key Components
- `main.py`: Encapsulates the core CLI application logic within the `RepoMapApp` class and serves as the main entry point.
- `file_scanner.py`: Handles scanning the repository, parsing `.gitignore`, and summarizing files.
- `code_parser.py`: Extracts structures like classes, functions, and imports from code files.
- `llm_service.py`: Manages interaction with the LLM for generating descriptions.
- `cache_manager.py`: Implements caching logic using SQLite to avoid reprocessing unchanged files.
- `config.py`: Manages application settings and API keys using Pydantic.
- `models.py`: Contains data models, including the list of supported languages.

## 📋 Additional Notes
- The tool supports a wide range of file types and programming languages. Check the `SUPPORTED_LANGUAGES` dictionary in the script for a full list.
- A pre-enhanced repository summary is saved to `.repo_map_structure.json` as an intermediate step.
- The tool uses a manual ignore list for generated files like `.repo_map_structure.json` and `.repo-map-cache.db`.
- SSL verification is handled using the certifi library for secure API communications.

## 🛡️ License
This project is licensed under the Apache 2.0 License. See the LICENSE file in the root directory of this project for the full license text.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/cyanheads/repo-map/issues).

## 📞 Support
If you encounter any problems or have any questions, please open an issue in the [GitHub repository](https://github.com/cyanheads/repo-map/issues).

## 📦 Version
Current version: 0.4.0

## ⚠️ Disclaimer
By using this tool, you acknowledge that files will be sent to the OpenRouter LLM for processing. Ensure you have the necessary permissions and consider any sensitive information in your repository.
