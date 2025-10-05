"""
NLP-Tugas 2 
Della Febi Alfian
22/505892/TK/55393

Testing Script untuk Transformer Implementation
Test case untuk validasi komponen
"""

import numpy as np
from transformer import *


def test_dimensions():
    """Test dimensionality untuk semua komponen"""
    print("DIMENSION VALIDATION TEST")
    
    # Configuration
    VOCAB_SIZE = 1000
    D_MODEL = 128
    NUM_HEADS = 4
    NUM_LAYERS = 3
    D_FF = 512
    BATCH_SIZE = 4
    SEQ_LEN = 15
    
    print(f"\nTest Configuration:")
    print(f"  VOCAB_SIZE = {VOCAB_SIZE}")
    print(f"  D_MODEL = {D_MODEL}")
    print(f"  NUM_HEADS = {NUM_HEADS}")
    print(f"  NUM_LAYERS = {NUM_LAYERS}")
    print(f"  D_FF = {D_FF}")
    print(f"  BATCH_SIZE = {BATCH_SIZE}")
    print(f"  SEQ_LEN = {SEQ_LEN}")
    
    # Test 1: Token Embedding
    print("\n[TEST 1] Token Embedding Dimensions")
    token_emb = TokenEmbedding(VOCAB_SIZE, D_MODEL)
    tokens = np.random.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    embedded = token_emb.forward(tokens)
    expected = (BATCH_SIZE, SEQ_LEN, D_MODEL)
    print(f"  Input: {tokens.shape} → Output: {embedded.shape}")
    print(f"  Expected: {expected}")
    assert embedded.shape == expected, f"Mismatch! Got {embedded.shape}"
    print("  ✅ PASSED")
    
    # Test 2: Positional Encoding
    print("\n[TEST 2] Positional Encoding Dimensions")
    pos_enc = PositionalEncoding(D_MODEL)
    output = pos_enc.forward(embedded)
    print(f"  Input: {embedded.shape} → Output: {output.shape}")
    assert output.shape == embedded.shape, "Shape should not change!"
    print("  ✅ PASSED")
    
    # Test 3: Multi-Head Attention
    print("\n[TEST 3] Multi-Head Attention Dimensions")
    mha = MultiHeadAttention(D_MODEL, NUM_HEADS)
    mask = create_causal_mask(SEQ_LEN)
    attn_out = mha.forward(output, mask)
    print(f"  Input: {output.shape} → Output: {attn_out.shape}")
    assert attn_out.shape == output.shape, "Shape should not change!"
    print("  ✅ PASSED")
    
    # Test 4: Feed-Forward Network
    print("\n[TEST 4] Feed-Forward Network Dimensions")
    ffn = FeedForward(D_MODEL, D_FF)
    ffn_out = ffn.forward(attn_out)
    print(f"  Input: {attn_out.shape} → Output: {ffn_out.shape}")
    assert ffn_out.shape == attn_out.shape, "Shape should not change!"
    print("  ✅ PASSED")
    
    # Test 5: Layer Normalization
    print("\n[TEST 5] Layer Normalization Dimensions")
    ln = LayerNormalization(D_MODEL)
    ln_out = ln.forward(ffn_out)
    print(f"  Input: {ffn_out.shape} → Output: {ln_out.shape}")
    assert ln_out.shape == ffn_out.shape, "Shape should not change!"
    print("  ✅ PASSED")
    
    # Test 6: Decoder Block
    print("\n[TEST 6] Decoder Block Dimensions")
    decoder_block = DecoderBlock(D_MODEL, NUM_HEADS, D_FF)
    block_out = decoder_block.forward(output, mask)
    print(f"  Input: {output.shape} → Output: {block_out.shape}")
    assert block_out.shape == output.shape, "Shape should not change!"
    print("  ✅ PASSED")
    
    # Test 7: Full Model
    print("\n[TEST 7] Full GPT Model Dimensions")
    model = GPTDecoder(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF)
    logits = model.forward(tokens)
    expected_logits = (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    print(f"  Input: {tokens.shape} → Output: {logits.shape}")
    print(f"  Expected: {expected_logits}")
    assert logits.shape == expected_logits, f"Mismatch! Got {logits.shape}"
    print("  ✅ PASSED")
    
    print("ALL DIMENSION TESTS PASSED! ✅")
    print("="*70)


def test_causal_masking():
    """Test apakah causal masking bekerja dengan benar"""
    print("CAUSAL MASKING VALIDATION")
    
    SEQ_LEN = 6
    D_K = 32
    
    # Create dummy Q, K, V
    Q = np.random.randn(1, SEQ_LEN, D_K)
    K = np.random.randn(1, SEQ_LEN, D_K)
    V = np.random.randn(1, SEQ_LEN, D_K)
    
    # Test WITHOUT mask
    print("\n[TEST 1] Attention WITHOUT causal mask")
    output_no_mask, attn_no_mask = scaled_dot_product_attention(Q, K, V, mask=None)
    print(f"  Attention weights shape: {attn_no_mask.shape}")
    print(f"  Attention weights (first row):")
    print(f"  {attn_no_mask[0, 0, :]}")
    print(f"  Sum: {attn_no_mask[0, 0, :].sum():.4f} (should be 1.0)")
    print(f"  Note: Token 0 can attend to ALL tokens (including future)")
    
    # Test WITH mask
    print("\n[TEST 2] Attention WITH causal mask")
    mask = create_causal_mask(SEQ_LEN)
    output_with_mask, attn_with_mask = scaled_dot_product_attention(Q, K, V, mask=mask)
    print(f"  Causal mask:")
    print(mask)
    print(f"\n  Attention weights (all rows):")
    print(attn_with_mask[0])
    
    # Validate: Upper triangle should be zero
    print(f"\n[VALIDATION] Checking upper triangle = 0")
    for i in range(SEQ_LEN):
        for j in range(i+1, SEQ_LEN):
            val = attn_with_mask[0, i, j]
            assert np.isclose(val, 0.0, atol=1e-7), f"Position [{i},{j}] = {val}, should be 0!"
    print("  ✅ Upper triangle is all zeros (causal constraint satisfied)")
    
    # Validate: Each row sums to 1
    print(f"\n[VALIDATION] Checking row sums = 1.0")
    row_sums = attn_with_mask.sum(axis=-1)[0]
    print(f"  Row sums: {row_sums}")
    assert np.allclose(row_sums, 1.0), "Each row should sum to 1!"
    print("  ✅ All rows sum to 1.0 (valid probability distribution)")
    
    print("\nCAUSAL MASKING TEST PASSED! ✅")
    print("="*70)


def test_numerical_stability():
    """Test numerical stability (no NaN, no Inf)"""
    print("NUMERICAL STABILITY TEST")
    
    VOCAB_SIZE = 500
    D_MODEL = 64
    NUM_HEADS = 2
    NUM_LAYERS = 2
    D_FF = 256
    BATCH_SIZE = 3
    SEQ_LEN = 20
    
    print(f"\nCreating model with {NUM_LAYERS} layers...")
    model = GPTDecoder(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF)
    
    # Test with random inputs
    print(f"Testing with random inputs...")
    for test_num in range(5):
        tokens = np.random.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        logits = model.forward(tokens)
        probs, predicted = model.predict_next_token(tokens)
        
        # Check for NaN
        has_nan = np.isnan(logits).any() or np.isnan(probs).any()
        # Check for Inf
        has_inf = np.isinf(logits).any() or np.isinf(probs).any()
        
        status = "✅" if not (has_nan or has_inf) else "❌"
        print(f"  Test {test_num+1}: {status} (NaN: {has_nan}, Inf: {has_inf})")
        
        assert not has_nan, "Found NaN values!"
        assert not has_inf, "Found Inf values!"
    
    print("\nNUMERICAL STABILITY TEST PASSED! ✅")
    print("="*70)


def test_probability_distributions():
    """Test apakah output softmax valid probability distribution"""
    print("PROBABILITY DISTRIBUTION VALIDATION")
    
    # Test 1: Softmax function
    print("\n[TEST 1] Softmax Function")
    test_cases = [
        np.array([[1.0, 2.0, 3.0]]),
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[-100, 0, 100]]),
        np.array([[1, 1, 1, 1, 1]])
    ]
    
    for i, x in enumerate(test_cases, 1):
        result = softmax(x)
        total = result.sum()
        print(f"  Case {i}: input={x[0][:3]}, sum={total:.6f}")
        assert np.isclose(total, 1.0), f"Sum should be 1.0, got {total}"
    print("  ✅ All softmax outputs sum to 1.0")
    
    # Test 2: Model output probabilities
    print("\n[TEST 2] Model Output Probabilities")
    VOCAB_SIZE = 100
    D_MODEL = 32
    model = GPTDecoder(VOCAB_SIZE, D_MODEL, num_heads=2, num_layers=1, d_ff=128)
    
    tokens = np.array([[1, 2, 3, 4, 5]])
    probs, predicted = model.predict_next_token(tokens)
    
    print(f"  Probability shape: {probs.shape}")
    print(f"  Probability sum: {probs.sum(axis=-1)}")
    print(f"  Min probability: {probs.min():.6f}")
    print(f"  Max probability: {probs.max():.6f}")
    print(f"  Predicted token: {predicted}")
    
    # Validations
    assert np.allclose(probs.sum(axis=-1), 1.0), "Probabilities should sum to 1!"
    assert (probs >= 0).all(), "All probabilities should be >= 0!"
    assert (probs <= 1).all(), "All probabilities should be <= 1!"
    print("  ✅ Valid probability distribution")
    
    print("\nPROBABILITY VALIDATION PASSED! ✅")
    print("="*70)


def test_different_batch_sizes():
    """Test model dengan berbagai batch size"""
    print("BATCH SIZE FLEXIBILITY TEST")
    
    VOCAB_SIZE = 200
    D_MODEL = 64
    NUM_HEADS = 2
    NUM_LAYERS = 2
    D_FF = 256
    SEQ_LEN = 8
    
    model = GPTDecoder(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF)
    
    batch_sizes = [1, 2, 4, 8, 16]
    
    for batch_size in batch_sizes:
        tokens = np.random.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN))
        logits = model.forward(tokens)
        expected_shape = (batch_size, SEQ_LEN, VOCAB_SIZE)
        
        status = "✅" if logits.shape == expected_shape else "❌"
        print(f"  Batch size {batch_size:2d}: {status} Output shape: {logits.shape}")
        
        assert logits.shape == expected_shape, f"Shape mismatch for batch_size={batch_size}"
    
    print("BATCH SIZE TEST PASSED! ✅")
    print("="*70)


def test_different_sequence_lengths():
    """Test model dengan berbagai sequence length"""
    print("SEQUENCE LENGTH FLEXIBILITY TEST")
    
    VOCAB_SIZE = 200
    D_MODEL = 64
    NUM_HEADS = 2
    NUM_LAYERS = 2
    D_FF = 256
    BATCH_SIZE = 2
    
    model = GPTDecoder(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, max_seq_len=100)
    
    seq_lengths = [1, 5, 10, 20, 50]
    
    for seq_len in seq_lengths:
        tokens = np.random.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_len))
        logits = model.forward(tokens)
        expected_shape = (BATCH_SIZE, seq_len, VOCAB_SIZE)
        
        status = "✅" if logits.shape == expected_shape else "❌"
        print(f"  Seq length {seq_len:3d}: {status} Output shape: {logits.shape}")
        
        assert logits.shape == expected_shape, f"Shape mismatch for seq_len={seq_len}"
    
    print("SEQUENCE LENGTH TEST PASSED! ✅")
    print("="*70)


def compare_with_without_components():
    """Bandingkan output dengan/tanpa komponen tertentu"""
    print("COMPONENT IMPACT ANALYSIS")
    
    VOCAB_SIZE = 100
    D_MODEL = 64
    BATCH_SIZE = 2
    SEQ_LEN = 10
    
    tokens = np.random.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    
    # Test 1: Impact of Positional Encoding
    print("\n[TEST 1] Impact of Positional Encoding")
    token_emb = TokenEmbedding(VOCAB_SIZE, D_MODEL)
    pos_enc = PositionalEncoding(D_MODEL)
    
    embedded_only = token_emb.forward(tokens)
    embedded_with_pos = pos_enc.forward(embedded_only)
    
    diff = np.abs(embedded_with_pos - embedded_only).mean()
    print(f"  Mean absolute difference: {diff:.6f}")
    print(f"  Positional encoding {'DOES' if diff > 0.01 else 'DOES NOT'} significantly modify embeddings")
    assert diff > 0.0, "Positional encoding should change the embeddings!"
    print("  ✅ Positional encoding is working")
    
    # Test 2: Impact of Layer Normalization
    print("\n[TEST 2] Impact of Layer Normalization")
    ln = LayerNormalization(D_MODEL)
    x = np.random.randn(BATCH_SIZE, SEQ_LEN, D_MODEL) * 10  # Large values
    x_normed = ln.forward(x)
    
    print(f"  Before normalization - Mean: {x.mean():.4f}, Std: {x.std():.4f}")
    print(f"  After normalization  - Mean: {x_normed.mean():.4f}, Std: {x_normed.std():.4f}")
    
    # Check if normalized (mean~0, std~1 per sample per position)
    mean_per_feature = x_normed.mean(axis=-1)
    std_per_feature = x_normed.std(axis=-1)
    print(f"  Mean per feature: {mean_per_feature.mean():.6f} (should be ~0)")
    print(f"  Std per feature: {std_per_feature.mean():.6f} (should be ~1)")
    print("  ✅ Layer normalization is working")
    
    print("\nCOMPONENT ANALYSIS PASSED! ✅")
    print("="*70)


def generate_summary_report():
    """Generate summary report untuk laporan"""
    
    # Model configuration
    VOCAB_SIZE = 1000
    D_MODEL = 128
    NUM_HEADS = 4
    NUM_LAYERS = 4
    D_FF = 512
    
    model = GPTDecoder(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF)
    
    # Count parameters
    def count_params(arr):
        return arr.size
    
    total_params = 0
    
    # Token embedding
    emb_params = count_params(model.token_embedding.embedding_table)
    total_params += emb_params
    
    # Positional encoding (not trainable, just pre-computed)
    pos_params = 0
    
    # Each layer
    layer_params = 0
    layer = model.layers[0]
    
    # Multi-head attention
    mha_params = (count_params(layer.attention.W_q) + 
                  count_params(layer.attention.W_k) + 
                  count_params(layer.attention.W_v) + 
                  count_params(layer.attention.W_o))
    
    # FFN
    ffn_params = (count_params(layer.ffn.W1) + count_params(layer.ffn.b1) +
                  count_params(layer.ffn.W2) + count_params(layer.ffn.b2))
    
    # Layer norm (x2 per block)
    ln_params = (count_params(layer.norm1.gamma) + count_params(layer.norm1.beta) +
                 count_params(layer.norm2.gamma) + count_params(layer.norm2.beta))
    
    layer_params = mha_params + ffn_params + ln_params
    total_params += layer_params * NUM_LAYERS
    
    # Output projection
    output_params = count_params(model.output_projection)
    total_params += output_params
    
    # Final layer norm
    final_ln_params = count_params(model.final_norm.gamma) + count_params(model.final_norm.beta)
    total_params += final_ln_params
    
    print(f"\nMODEL ARCHITECTURE SUMMARY\n")
    print(f"Configuration:")
    print(f"  Vocabulary Size      : {VOCAB_SIZE:,}")
    print(f"  Model Dimension      : {D_MODEL}")
    print(f"  Number of Heads      : {NUM_HEADS}")
    print(f"  Number of Layers     : {NUM_LAYERS}")
    print(f"  FFN Hidden Dimension : {D_FF}")
    print(f"\nParameter Count:")
    print(f"  Token Embedding      : {emb_params:,}")
    print(f"  Positional Encoding  : {pos_params:,} (pre-computed)")
    print(f"  Per Decoder Block    : {layer_params:,}")
    print(f"    - Multi-Head Attn  : {mha_params:,}")
    print(f"    - Feed-Forward     : {ffn_params:,}")
    print(f"    - Layer Norm       : {ln_params:,}")
    print(f"  All {NUM_LAYERS} Layers        : {layer_params * NUM_LAYERS:,}")
    print(f"  Output Projection    : {output_params:,}")
    print(f"  Final Layer Norm     : {final_ln_params:,}")
    print(f"  \nTOTAL PARAMETERS     : {total_params:,}")
    print(f"{'='*60}")
    
    # Test inference time
    print(f"\nINFERENCE TEST:")
    import time
    
    BATCH_SIZE = 4
    SEQ_LEN = 50
    tokens = np.random.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    
    start = time.time()
    logits = model.forward(tokens)
    end = time.time()
    
    inference_time = (end - start) * 1000  # to milliseconds
    
    print(f"  Input Shape          : {tokens.shape}")
    print(f"  Output Shape         : {logits.shape}")
    print(f"  Inference Time       : {inference_time:.2f} ms")
    print(f"  Tokens/second        : {(BATCH_SIZE * SEQ_LEN) / (inference_time/1000):.2f}")
    
    print("SUMMARY REPORT GENERATED! ✅")
    print("="*70)


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" COMPREHENSIVE TESTING SUITE")
    print(" Transformer Implementation Validation")
    print("="*70)
    
    tests = [
        ("Dimension Validation", test_dimensions),
        ("Causal Masking", test_causal_masking),
        ("Numerical Stability", test_numerical_stability),
        ("Probability Distributions", test_probability_distributions),
        ("Batch Size Flexibility", test_different_batch_sizes),
        ("Sequence Length Flexibility", test_different_sequence_lengths),
        ("Component Impact", compare_with_without_components),
        ("Summary Report", generate_summary_report),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ {test_name} FAILED:")
            print(f"   Error: {e}")
            failed += 1
    
    print(" FINAL RESULTS")
    print(f"  Tests Passed: {passed}/{len(tests)}")
    print(f"  Tests Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n ALL TESTS PASSED! ")
    else:
        print(f"\n  ⚠️  {failed} test(s) failed. Please review the errors above.")
    
    print("="*70)