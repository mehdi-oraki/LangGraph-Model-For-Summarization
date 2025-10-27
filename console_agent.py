"""
Console Interface for Simple AI Agent
A clean, user-friendly console interface for the text summarization agent.
"""

from simple_ai_agent import SimpleAIAgent
import sys

def print_header():
    """Print application header"""
    print("=" * 60)
    print("🤖 Simple AI Agent - Text Summarization")
    print("🔗 Powered by LangGraph & Hugging Face")
    print("=" * 60)

def print_help():
    """Print help information"""
    print("\n📖 Available Commands:")
    print("  help, h     - Show this help message")
    print("  quit, q     - Exit the application")
    print("  clear, c    - Clear the screen")
    print("  status      - Show agent status")
    print("\n💡 Just type any text to summarize it!")

def print_status(agent):
    """Print agent status"""
    print(f"\n📊 Agent Status:")
    print(f"  Model: {agent.model_name}")
    print(f"  Device: {agent.device}")
    print(f"  Model Loaded: {'Yes' if agent.model is not None else 'No'}")

def clear_screen():
    """Clear the console screen"""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    """Main console application"""
    print_header()
    
    # Initialize agent
    try:
        agent = SimpleAIAgent()
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        sys.exit(1)
    
    print_help()
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            user_input = input("\n🤖 Agent> ").strip()
            
            # Handle commands
            if user_input.lower() in ['quit', 'q', 'exit']:
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() in ['help', 'h']:
                print_help()
                continue
            
            elif user_input.lower() in ['clear', 'c']:
                clear_screen()
                print_header()
                continue
            
            elif user_input.lower() == 'status':
                print_status(agent)
                continue
            
            elif not user_input:
                print("⚠️ Please enter some text or a command.")
                continue
            
            # Process text
            print("\n🔄 Processing...")
            result = agent.process_text(user_input)
            
            if result["success"]:
                print(f"\n✅ Summarized Translation: {result['summarized_text']}")
            else:
                print(f"\n❌ Error: {result['error']}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()