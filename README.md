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
- 🔒 Respects .gitignore files (including nested ones) within the target directory for file exclusion
- 🚦 Implements rate limiting and exponential backoff for LLM API calls
- ⚡ Asynchronous processing for improved performance

## 🛠️ Installation
Install repo-map using pip:
```bash
pip install repo-map
```
Additionally, the `repo_map.py` file is designed to be a standalone script that can be easily integrated into any existing project repository.

## 🚀 Usage
To generate a repository map, run the following command:
```bash
repo-map <repository_path> [options]
```
Replace `<repository_path>` with the path to the repository you want to analyze.

### Options:
- `-y,` `--yes`: Automatically accept the disclaimer and proceed without prompting
- `--model MODEL`: Specify the OpenRouter LLM model to use (default: anthropic/claude-3.5-sonnet)

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
│   ├── Description: Entry point for the Snake game, initializes game and runs the main loop.
│   ├── Developer Consideration: "Uses pygame for game development, which may require additional setup for cross-platform compatibility."
│   ├── Imports: [pygame, random, time]
│   ├── Functions: [main, game_loop, draw_snake, draw_food]
├── config.py (Python)
│   ├── Description: Centralizes game configuration parameters.
│   ├── Developer Consideration: "Hard-coded values may need adjustment for different screen sizes or game difficulties."
├── assets/
│   ├── images/
│   │   ├── snake_head.png (Image)
│   │   ├── food.png (Image)
│   ├── sounds/
│   │   ├── eat.wav (Audio)
│   │   ├── game_over.mp3 (Audio)
├── requirements.txt (Text)
│   ├── Description: Lists all Python package dependencies for the project.
├── README.md (Markdown)
│   ├── Description: Provides project overview, setup instructions, and gameplay details.
└──────────────
```

This example demonstrates how repo-map provides a comprehensive overview of a Snake game project, including file descriptions, developer considerations, and key structural information.

## 🔧 Requirements
- Python 3.7+
- Dependencies:
  - requests
  - tqdm
  - aiohttp
  - pathspec
  - certifi

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
- `summarize_repo()`: Generates the initial repository summary
- `enhance_repo_with_llm()`: Enhances the summary with AI-generated descriptions
- `get_structure()`: Extracts classes, functions, and constants from files
- `print_tree()`: Displays the repository structure in a tree format
- `save_tree_map()`: Saves the repository map to a Markdown file

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
Current version: 0.1.0

## ⚠️ Disclaimer
By using this tool, you acknowledge that files will be sent to the OpenRouter LLM for processing. Ensure you have the necessary permissions and consider any sensitive information in your repository.