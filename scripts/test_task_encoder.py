"""
Test TaskEncoder components to prove they work.

Tests both Version A (CNN) and Version B (ContextEncoder) with dummy data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.models.task_encoder_cnn import TaskEncoderCNN
from src.models.task_encoder_advanced import TaskEncoderAdvanced

print("="*70)
print("TaskEncoder Component Tests")
print("="*70)

# Test parameters
batch_size = 4
num_demos = 3
H, W = 30, 30  # Grid size
embed_dim = 256

# Create dummy demonstration data
demo_input = torch.randint(0, 11, (batch_size, num_demos, H, W))
demo_output = torch.randint(0, 11, (batch_size, num_demos, H, W))

print(f"\nTest Data:")
print(f"  Batch size: {batch_size}")
print(f"  Num demos: {num_demos}")
print(f"  Grid size: {H}x{W}")
print(f"  Demo input shape: {demo_input.shape}")
print(f"  Demo output shape: {demo_output.shape}")

# Test Version A: TaskEncoderCNN
print(f"\n{'='*70}")
print("Test 1: TaskEncoderCNN (Version A - Simple CNN)")
print("="*70)

try:
    model_cnn = TaskEncoderCNN(embed_dim=embed_dim, num_demos=num_demos)
    num_params = sum(p.numel() for p in model_cnn.parameters())
    print(f"  ✅ Model created: {num_params:,} parameters")
    
    # Test forward pass (inference mode)
    with torch.no_grad():
        embeddings_eval = model_cnn(demo_input, demo_output)
    
    print(f"  ✅ Forward pass works (inference)")
    print(f"     Input: {demo_input.shape}")
    print(f"     Output: {embeddings_eval.shape}")
    assert embeddings_eval.shape == (batch_size, embed_dim), f"Expected ({batch_size}, {embed_dim}), got {embeddings_eval.shape}"
    print(f"  ✅ Output shape correct: ({batch_size}, {embed_dim})")
    
    # Test backward pass (training mode)
    embeddings_train = model_cnn(demo_input, demo_output)
    loss = embeddings_train.mean()
    loss.backward()
    print(f"  ✅ Backward pass works (training)")
    
    print(f"\n  🎉 TaskEncoderCNN (Version A) works!")
    
except Exception as e:
    print(f"  ❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test Version B: TaskEncoderAdvanced
print(f"\n{'='*70}")
print("Test 2: TaskEncoderAdvanced (Version B - Champion's ContextEncoder)")
print("="*70)

try:
    model_advanced = TaskEncoderAdvanced(embed_dim=embed_dim, num_demos=num_demos)
    num_params = sum(p.numel() for p in model_advanced.parameters())
    print(f"  ✅ Model created: {num_params:,} parameters")
    
    # Test forward pass (inference mode)
    with torch.no_grad():
        embeddings_eval = model_advanced(demo_input, demo_output)
    
    print(f"  ✅ Forward pass works (inference)")
    print(f"     Input: {demo_input.shape}")
    print(f"     Output: {embeddings_eval.shape}")
    assert embeddings_eval.shape == (batch_size, embed_dim), f"Expected ({batch_size}, {embed_dim}), got {embeddings_eval.shape}"
    print(f"  ✅ Output shape correct: ({batch_size}, {embed_dim})")
    
    # Test backward pass (training mode)
    embeddings_train = model_advanced(demo_input, demo_output)
    loss = embeddings_train.mean()
    loss.backward()
    print(f"  ✅ Backward pass works (training)")
    
    print(f"\n  🎉 TaskEncoderAdvanced (Version B) works!")
    
except Exception as e:
    print(f"  ❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with different embed_dim (Phase 2 uses 400)
print(f"\n{'='*70}")
print("Test 3: Both models with embed_dim=400 (Phase 2)")
print("="*70)

try:
    model_cnn_p2 = TaskEncoderCNN(embed_dim=400, num_demos=num_demos)
    model_adv_p2 = TaskEncoderAdvanced(embed_dim=400, num_demos=num_demos)
    
    with torch.no_grad():
        emb_cnn = model_cnn_p2(demo_input, demo_output)
        emb_adv = model_adv_p2(demo_input, demo_output)
    
    assert emb_cnn.shape == (batch_size, 400)
    assert emb_adv.shape == (batch_size, 400)
    
    print(f"  ✅ CNN output: {emb_cnn.shape}")
    print(f"  ✅ Advanced output: {emb_adv.shape}")
    print(f"\n  🎉 Both models work with Phase 2 dimensions!")
    
except Exception as e:
    print(f"  ❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test classification head
print(f"\n{'='*70}")
print("Test 4: Classification Head (Direct Classification - Phase 1)")
print("="*70)

try:
    num_categories = 9  # S1-S3, C1-C2, K1, L1, A1-A2
    
    # Create encoder + classifier
    encoder = TaskEncoderCNN(embed_dim=256)
    classifier = nn.Linear(256, num_categories)
    
    with torch.no_grad():
        embeddings = encoder(demo_input, demo_output)
        logits = classifier(embeddings)
    
    print(f"  ✅ Encoder output: {embeddings.shape}")
    print(f"  ✅ Classifier logits: {logits.shape}")
    assert logits.shape == (batch_size, num_categories)
    
    # Test with softmax
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)
    
    print(f"  ✅ Probabilities: {probs.shape}")
    print(f"  ✅ Predictions: {preds.shape}")
    print(f"     Sample predictions: {preds.tolist()}")
    
    print(f"\n  🎉 Classification pipeline works!")
    
except Exception as e:
    print(f"  ❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print(f"\n{'='*70}")
print("🎉 ALL TESTS PASSED")
print("="*70)
print(f"\nComponents verified:")
print(f"  ✅ TaskEncoderCNN (Version A) - {sum(p.numel() for p in TaskEncoderCNN().parameters()):,} params")
print(f"  ✅ TaskEncoderAdvanced (Version B) - {sum(p.numel() for p in TaskEncoderAdvanced().parameters()):,} params")
print(f"  ✅ Forward/backward passes work")
print(f"  ✅ Phase 1 dimensions (256) work")
print(f"  ✅ Phase 2 dimensions (400) work")
print(f"  ✅ Classification head works")
print(f"\nReady for:")
print(f"  1. Phase 1C: Direct classification training")
print(f"  2. Phase 2B: Oracle-predictor training")
print(f"="*70)
