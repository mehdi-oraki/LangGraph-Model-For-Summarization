"""
Simple AI Agent using LangGraph and Local Hugging Face Model
A minimalist text summarization agent with state management.
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Define the state structure for our LangGraph agent
class AgentState(TypedDict):
    """State structure for the text summarization agent"""
    user_input: str           # Original user text
    summarized_text: str      # Generated summary
    model_loaded: bool        # Whether model is loaded
    error_message: str        # Any error messages
    processing_status: str    # Current processing status

class SimpleAIAgent:
    """Simple AI Agent for text summarization using local Hugging Face model"""
    
    def __init__(self, model_name: str = "google/flan-t5-small"):
        """
        Initialize the AI agent with a local Hugging Face model
        
        Args:
            model_name: Name of the Hugging Face model to use
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🤖 Initializing Simple AI Agent...")
        print(f"📱 Using device: {self.device}")
        print(f"🧠 Model: {model_name}")
    
    def load_model_node(self, state: AgentState) -> AgentState:
        """
        LangGraph node: Load the Hugging Face model and tokenizer
        """
        print("📦 Loading Hugging Face model...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print(f"✅ Tokenizer loaded: {self.model_name}")
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            print(f"✅ Model loaded and moved to {self.device}")
            
            state["model_loaded"] = True
            state["processing_status"] = "Model loaded successfully"
            state["error_message"] = ""
            
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            print(f"❌ {error_msg}")
            state["model_loaded"] = False
            state["processing_status"] = "Model loading failed"
            state["error_message"] = error_msg
        
        return state
    
    def summarize_text_node(self, state: AgentState) -> AgentState:
        """
        LangGraph node: Generate summary of the input text
        """
        print("🔄 Processing text summarization...")
        
        if not state.get("model_loaded", False):
            state["error_message"] = "Model not loaded. Cannot process text."
            state["processing_status"] = "Processing failed"
            return state
        
        user_input = state.get("user_input", "")
        if not user_input.strip():
            state["error_message"] = "No input text provided."
            state["processing_status"] = "Processing failed"
            return state
        
        try:
            # Create summarization prompt
            prompt = f"Summarize the following text in a concise way: {user_input}"
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate summary
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=150,  # Limit summary length
                    num_beams=4,     # Beam search for better quality
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode the generated text
            summarized_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the output
            summarized_text = summarized_text.strip()
            
            state["summarized_text"] = summarized_text
            state["processing_status"] = "Summarization completed"
            state["error_message"] = ""
            
            print(f"✅ Summarization completed")
            print(f"📝 Original length: {len(user_input)} characters")
            print(f"📝 Summary length: {len(summarized_text)} characters")
            
        except Exception as e:
            error_msg = f"Summarization failed: {str(e)}"
            print(f"❌ {error_msg}")
            state["error_message"] = error_msg
            state["processing_status"] = "Summarization failed"
            state["summarized_text"] = ""
        
        return state
    
    def create_workflow(self) -> StateGraph:
        """
        Create and configure the LangGraph workflow
        """
        print("🔧 Creating LangGraph workflow...")
        
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes to the workflow
        workflow.add_node("load_model", self.load_model_node)
        workflow.add_node("summarize_text", self.summarize_text_node)
        
        # Define the execution flow
        workflow.set_entry_point("load_model")
        workflow.add_edge("load_model", "summarize_text")
        workflow.add_edge("summarize_text", END)
        
        # Compile the graph
        agent = workflow.compile()
        
        print("✅ LangGraph workflow created successfully")
        return agent
    
    def process_text(self, user_input: str) -> dict:
        """
        Process user input through the LangGraph agent
        
        Args:
            user_input: Text to be summarized
            
        Returns:
            Dictionary containing the results
        """
        print(f"\n🚀 Processing text: '{user_input[:50]}{'...' if len(user_input) > 50 else ''}'")
        
        # Create the workflow
        agent = self.create_workflow()
        
        # Initialize state
        initial_state = {
            "user_input": user_input,
            "summarized_text": "",
            "model_loaded": False,
            "error_message": "",
            "processing_status": "Starting..."
        }
        
        # Run the agent
        try:
            result = agent.invoke(initial_state)
            
            # Prepare output
            output = {
                "success": result.get("error_message", "") == "",
                "original_text": result.get("user_input", ""),
                "summarized_text": result.get("summarized_text", ""),
                "status": result.get("processing_status", ""),
                "error": result.get("error_message", "")
            }
            
            return output
            
        except Exception as e:
            return {
                "success": False,
                "original_text": user_input,
                "summarized_text": "",
                "status": "Agent execution failed",
                "error": str(e)
            }

def main():
    """
    Main function to demonstrate the AI agent
    """
    print("=" * 60)
    print("🤖 Simple AI Agent - Text Summarization Demo")
    print("=" * 60)
    
    # Initialize the agent
    agent = SimpleAIAgent()
    
    # Example texts for demonstration
    example_texts = [
        "Artificial intelligence is transforming the way we work and live. From healthcare to transportation, AI technologies are being integrated into various industries to improve efficiency and solve complex problems. Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions that were previously impossible for humans to achieve.",
        
        "Climate change represents one of the most pressing challenges of our time. Rising global temperatures, melting ice caps, and extreme weather events are all indicators of a changing planet. Scientists around the world are working to understand these changes and develop solutions to mitigate their impact on ecosystems and human societies.",
        
        "The development of renewable energy sources has accelerated significantly in recent years. Solar panels and wind turbines are becoming more efficient and cost-effective, making clean energy accessible to more people worldwide. This shift towards sustainable energy is crucial for reducing our dependence on fossil fuels and combating climate change."
    ]
    
    # Process each example
    for i, text in enumerate(example_texts, 1):
        print(f"\n📝 Example {i}:")
        print("-" * 40)
        
        result = agent.process_text(text)
        
        if result["success"]:
            print(f"✅ Status: {result['status']}")
            print(f"📄 Original Text: {result['original_text']}")
            print(f"📝 Summarized Translation: {result['summarized_text']}")
        else:
            print(f"❌ Error: {result['error']}")
        
        print("-" * 40)
    
    # Interactive mode
    print(f"\n🎯 Interactive Mode - Enter your own text to summarize:")
    print("(Type 'quit' to exit)")
    
    while True:
        try:
            user_input = input("\n📝 Enter text to summarize: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                print("⚠️ Please enter some text to summarize.")
                continue
            
            result = agent.process_text(user_input)
            
            if result["success"]:
                print(f"\n✅ Status: {result['status']}")
                print(f"📝 Summarized Translation: {result['summarized_text']}")
            else:
                print(f"\n❌ Error: {result['error']}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()