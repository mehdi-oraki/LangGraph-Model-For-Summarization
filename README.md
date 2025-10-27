<div align="center">

# 🤖 Simple AI Agent

**A minimalist text summarization agent using LangGraph and local Hugging Face models**

*Clean, efficient, and educational implementation*

</div>

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
