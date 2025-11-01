<div align="center">

# 🤖 Simple AI Agent

**A minimalist text summarization agent using LangGraph and local Hugging Face models**

*Clean, efficient, and educational implementation*

</div>

---

## 📑 Table of Contents

- [📋 Overview](#-overview)
- [🚀 Features](#-features)
- [🛠️ Technical Architecture](#️-technical-architecture)
  - [Core Components](#core-components)
- [📁 Project Structure](#-project-structure)
- [⚙️ Installation](#️-installation)
- [🎮 Usage](#-usage)
- [🧪 Testing](#-testing)
- [📦 Dependencies](#-dependencies)
- [🤝 Contributing](#-contributing)
- [📝 License](#-license)

---

## 📋 Overview

This project demonstrates how to build a simple AI agent using LangGraph for state management and a local Hugging Face model for text summarization. The agent is designed to be educational, showing core LangGraph concepts while maintaining simplicity.

## 🚀 Features

- **LangGraph Orchestration**: Stateful execution flow with nodes and edges
- **Local Hugging Face Model**: Uses `google/flan-t5-small` for text summarization
- **Minimalist Design**: Clean, focused implementation for learning
- **Console Interface**: Simple command-line interaction
- **Error Handling**: Robust error management throughout the workflow

## 🛠️ Technical Architecture

### Core Components

1. **LangGraph State Management**
   - `AgentState`: TypedDict defining the agent's state structure
   - Stateful execution flow: Model Loading → Text Summarization → End

2. **Hugging Face Integration**
   - Local model loading and inference
   - Tokenization and text processing
   - Device optimization (CPU/CUDA)

3. **Workflow Nodes**
   - `load_model_node()`: Loads Hugging Face model and tokenizer
   - `summarize_text_node()`: Processes text and generates summaries

## 📁 Project Structure

```
agent2-test/
├── simple_ai_agent.py      # Core agent implementation with LangGraph
├── console_agent.py         # Console interface for interactive use
├── check_huggingface.py     # Verification script for HF setup
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

### File Descriptions

- **`simple_ai_agent.py`**: Main agent class with LangGraph orchestration, Hugging Face model integration, and text summarization logic
- **`console_agent.py`**: User-friendly console interface with commands (help, quit, status, clear)
- **`check_huggingface.py`**: Utility script to verify Hugging Face installation and GPU availability
- **`requirements.txt`**: List of required Python packages

## ⚙️ Installation

1. **Clone or navigate to the project directory**

2. **Create a virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
python check_huggingface.py
```

This will download the `google/flan-t5-small` model on first run (approximately 280 MB).

## 🎮 Usage

### Interactive Console

Run the console agent for an interactive experience:

```bash
python console_agent.py
```

**Available Commands:**
- `help` or `h` - Show available commands
- `quit` or `q` - Exit the application
- `clear` or `c` - Clear the screen
- `status` - Display agent status (model, device, etc.)
- Any other text - Summarize it!

### Programmatic Usage

```python
from simple_ai_agent import SimpleAIAgent

# Initialize agent
agent = SimpleAIAgent()

# Process text
result = agent.process_text("Your long text here...")

if result["success"]:
    print(f"Summary: {result['summarized_text']}")
else:
    print(f"Error: {result['error']}")
```

## 🧪 Testing

Run the verification script to check your setup:

```bash
python check_huggingface.py
```

Expected output:
- ✅ PyTorch installation
- ✅ CUDA availability (if GPU present)
- ✅ Model and tokenizer loading
- ✅ Sample inference

## 📦 Dependencies

- **langgraph** (≥0.2.0): Orchestration framework for stateful AI workflows
- **transformers** (≥4.30.0): Hugging Face transformers library
- **torch** (≥2.0.0): PyTorch deep learning framework
- **accelerate** (≥0.20.0): Hugging Face acceleration library
- **sentencepiece** (≥0.1.99): Tokenization library
- **protobuf** (≥3.20.0): Protocol buffer support

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with different models
- Enhance the agent capabilities
- Improve error handling
- Add new features

## 📝 License

This project is for educational purposes. Please respect the licenses of:
- LangGraph
- Hugging Face Transformers
- PyTorch
- The underlying `google/flan-t5-small` model
