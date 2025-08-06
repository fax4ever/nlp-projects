#!/usr/bin/env python3
"""
Simple example of how to use the LLM sentence splitter.
"""

from generated_code.llm_sentence_splitter import LLMSentenceSplitter


def quick_test():
    """Quick test with a pre-trained model (no fine-tuning)."""
    
    print("🚀 Testing LLM Sentence Splitter")
    print("=" * 50)
    
    # Initialize with a smaller, faster model for testing
    splitter = LLMSentenceSplitter("microsoft/Phi-3.5-mini-instruct")
    
    # Test text (Italian)
    test_text = ("Quel ramo del lago di Como che volge a mezzogiorno tra due catene "
                "non interrotte di monti tutto a seni e a golfi a seconda dello "
                "sporgere e del rientrare di quelli vien quasi a un tratto a "
                "ristringersi e a prender corso e figura di fiume")
    
    print(f"Original text:\n{test_text}\n")
    
    try:
        # Get sentence splits
        sentences = splitter.inference(test_text)
        
        print(f"Detected {len(sentences)} sentences:")
        for i, sentence in enumerate(sentences, 1):
            print(f"{i}. {sentence}")
            
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Note: This requires GPU and the necessary libraries installed.")
        print("Install with: pip install -r requirements_llm.txt")


def training_example():
    """Example of how to run training."""
    
    print("\n🔥 Training Example")
    print("=" * 50)
    print("To train the model, uncomment and run the following:")
    print()
    print("# splitter = LLMSentenceSplitter('microsoft/Phi-3.5-mini-instruct')")
    print("# splitter.train(")
    print("#     output_dir='./italian-sentence-splitter-llm',")
    print("#     num_epochs=2")
    print("# )")
    print()
    print("Note: Training requires:")
    print("- GPU with at least 8GB VRAM")
    print("- HF_TOKEN environment variable set")
    print("- About 1-2 hours depending on your hardware")


if __name__ == "__main__":
    quick_test()
    training_example()