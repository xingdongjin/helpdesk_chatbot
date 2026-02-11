#!/usr/bin/env python3
"""
Quick test to verify Intel GPU (XPU) acceleration is working with the vector store.
"""

import os
import sys
import torch
import time

# Add parent directory to Python path so imports work correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.vector_store import VectorStore

def test_gpu_detection():
    """Test GPU detection and availability."""
    print("="*70)
    print("GPU Detection Test")
    print("="*70)

    # Check if XPU is available
    if torch.xpu.is_available():
        print(f"✓ Intel GPU (XPU) is available")
        print(f"  Device: {torch.xpu.get_device_name(0)}")
        print(f"  Device count: {torch.xpu.device_count()}")
        return True
    else:
        print("✗ Intel GPU (XPU) is NOT available")
        return False


def test_vector_store_performance():
    """Test vector store with GPU and measure performance."""
    print("\n" + "="*70)
    print("Vector Store GPU Performance Test")
    print("="*70)

    # Initialize vector store (should auto-detect GPU)
    print("\nInitializing vector store...")
    vector_store = VectorStore()

    # Prepare test data
    test_texts = [
        "The AI plush toy is perfect for children aged 4-8 years old.",
        "Buddy Bear costs $89.99 and comes with a charging cable.",
        "Our toys feature advanced AI technology and voice recognition.",
        "The battery lasts up to 8 hours on a single charge.",
        "All toys are washable with removable electronic components.",
    ] * 20  # Duplicate to have more data for testing

    print(f"\nTesting with {len(test_texts)} sample texts...")
    print(f"Using device: {vector_store.device}")

    # Benchmark encoding speed
    print("\nGenerating embeddings...")
    start_time = time.time()

    embeddings = vector_store.model.encode(
        test_texts,
        show_progress_bar=True,
        device=vector_store.device
    )

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"\n✓ Generated {len(test_texts)} embeddings in {elapsed:.2f} seconds")
    print(f"  Speed: {len(test_texts)/elapsed:.1f} embeddings/second")

    if vector_store.device == 'xpu':
        print(f"  🚀 Using Intel GPU acceleration!")
    else:
        print(f"  💻 Using CPU (consider enabling GPU for faster performance)")

    return True


def main():
    """Run all GPU tests."""
    print("\nIntel GPU Verification for FluffyAI Helpdesk Chatbot\n")

    # Test 1: GPU Detection
    gpu_available = test_gpu_detection()

    # Test 2: Vector Store Performance
    test_vector_store_performance()

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    if gpu_available:
        print("\n✓ GPU acceleration is enabled and working!")
        print("  Your embeddings will be generated much faster.")
    else:
        print("\n⚠ GPU acceleration is not available.")
        print("  The system will use CPU, which is slower but still functional.")

    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nError during testing: {e}")
        import traceback
        traceback.print_exc()
