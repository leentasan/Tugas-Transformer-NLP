"""
NLP-Tugas 2 
Della Febi Alfian
22/606892/TK/55393

Implementasi Transformer (GPT-style Decoder) from Scratch dengan NumPy
Tugas Individu NLP - Arsitektur Transformer

Komponen:
1. Token Embedding
2. Positional Encoding (sinusoidal)
3. Scaled Dot-Product Attention dengan softmax
4. Multi-Head Attention
5. Feed-Forward Network (FFN)
6. Residual Connection + Layer Normalization (pre-norm)
7. Causal Masking
8. Output Layer dengan softmax
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def softmax(x, axis=-1):
    """
    Softmax function yang numerically stable
    
    Args:
        x: input array
        axis: axis untuk softmax (default=-1)
    
    Returns:
        Probability distribution
    """
    # Trick: kurangi dengan max untuk stabilitas numerik
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x):
    """
    Gaussian Error Linear Unit (GELU) activation function
    Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    
    Args:
        x: input array
    
    Returns:
        Activated output
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def create_causal_mask(seq_len):
    """
    Membuat causal mask untuk mencegah attention melihat token masa depan
    
    Args:
        seq_len: panjang sequence
    
    Returns:
        mask shape (seq_len, seq_len) dengan nilai 0 dan -inf
        
    Example untuk seq_len=4:
        [[   0, -inf, -inf, -inf],
         [   0,    0, -inf, -inf],
         [   0,    0,    0, -inf],
         [   0,    0,    0,    0]]
    """
    mask = np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1)
    return mask


# ============================================================================
# KOMPONEN 1: TOKEN EMBEDDING
# ============================================================================

class TokenEmbedding:
    """
    Mengubah token ID menjadi vektor embedding
    """
    def __init__(self, vocab_size, d_model):
        """
        Args:
            vocab_size: ukuran vocabulary
            d_model: dimensi embedding
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Inisialisasi embedding table dengan distribusi normal kecil
        self.embedding_table = np.random.randn(vocab_size, d_model) * 0.01
    
    def forward(self, token_ids):
        """
        Args:
            token_ids: array of token IDs, shape (batch_size, seq_len)
        
        Returns:
            embeddings: shape (batch_size, seq_len, d_model)
        """
        return self.embedding_table[token_ids]


# ============================================================================
# KOMPONEN 2: POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding:
    """
    Sinusoidal Positional Encoding untuk memberikan informasi posisi token
    """
    def __init__(self, d_model, max_seq_len=512):
        """
        Args:
            d_model: dimensi embedding
            max_seq_len: panjang maksimal sequence yang didukung
        """
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Pre-compute positional encoding
        pe = np.zeros((max_seq_len, d_model))
        position = np.arange(0, max_seq_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe
    
    def forward(self, x):
        """
        Args:
            x: input embeddings, shape (batch_size, seq_len, d_model)
        
        Returns:
            x + positional encoding, shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        # Ambil positional encoding sesuai panjang sequence
        pos_encoding = self.pe[:seq_len, :]
        # Tambahkan ke input (broadcasting untuk batch)
        return x + pos_encoding


# ============================================================================
# KOMPONEN 3: SCALED DOT-PRODUCT ATTENTION
# ============================================================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Implementasi Scaled Dot-Product Attention
    
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        Q: Query, shape (batch, seq_len, d_k)
        K: Key, shape (batch, seq_len, d_k)
        V: Value, shape (batch, seq_len, d_v)
        mask: Optional causal mask, shape (seq_len, seq_len)
    
    Returns:
        output: shape (batch, seq_len, d_v)
        attention_weights: shape (batch, seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    
    # 1. Compute Q @ K^T
    # (batch, seq_len, d_k) @ (batch, d_k, seq_len) -> (batch, seq_len, seq_len)
    scores = Q @ K.transpose(0, 2, 1)
    
    # 2. Scale by sqrt(d_k)
    scores = scores / np.sqrt(d_k)
    
    # 3. Apply mask (if provided)
    if mask is not None:
        scores = scores + mask  # mask berisi 0 dan -inf
    
    # 4. Softmax
    attention_weights = softmax(scores, axis=-1)
    
    # 5. Multiply with V
    # (batch, seq_len, seq_len) @ (batch, seq_len, d_v) -> (batch, seq_len, d_v)
    output = attention_weights @ V
    
    return output, attention_weights


# ============================================================================
# KOMPONEN 4: MULTI-HEAD ATTENTION
# ============================================================================

class MultiHeadAttention:
    """
    Multi-Head Attention mechanism
    """
    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model: dimensi model (harus habis dibagi num_heads)
            num_heads: jumlah attention heads
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimensi per head
        
        # Weight matrices untuk Q, K, V
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        
        # Output projection
        self.W_o = np.random.randn(d_model, d_model) * 0.01
    
    def split_heads(self, x):
        """
        Split tensor menjadi multiple heads
        
        Args:
            x: shape (batch, seq_len, d_model)
        
        Returns:
            shape (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.shape
        # Reshape: (batch, seq_len, num_heads, d_k)
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        # Transpose: (batch, num_heads, seq_len, d_k)
        return x.transpose(0, 2, 1, 3)
    
    def combine_heads(self, x):
        """
        Combine multiple heads kembali
        
        Args:
            x: shape (batch, num_heads, seq_len, d_k)
        
        Returns:
            shape (batch, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.shape
        # Transpose: (batch, seq_len, num_heads, d_k)
        x = x.transpose(0, 2, 1, 3)
        # Reshape: (batch, seq_len, d_model)
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def forward(self, x, mask=None):
        """
        Forward pass Multi-Head Attention
        
        Args:
            x: input, shape (batch, seq_len, d_model)
            mask: optional causal mask
        
        Returns:
            output: shape (batch, seq_len, d_model)
        """
        batch_size = x.shape[0]
        
        # 1. Linear projections untuk Q, K, V
        Q = x @ self.W_q  # (batch, seq_len, d_model)
        K = x @ self.W_k
        V = x @ self.W_v
        
        # 2. Split menjadi multiple heads
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # 3. Scaled dot-product attention untuk setiap head
        # Reshape untuk batch processing: (batch * num_heads, seq_len, d_k)
        Q_reshaped = Q.reshape(batch_size * self.num_heads, Q.shape[2], self.d_k)
        K_reshaped = K.reshape(batch_size * self.num_heads, K.shape[2], self.d_k)
        V_reshaped = V.reshape(batch_size * self.num_heads, V.shape[2], self.d_k)
        
        attn_output, _ = scaled_dot_product_attention(Q_reshaped, K_reshaped, V_reshaped, mask)
        
        # Reshape kembali: (batch, num_heads, seq_len, d_k)
        attn_output = attn_output.reshape(batch_size, self.num_heads, attn_output.shape[1], self.d_k)
        
        # 4. Combine heads
        combined = self.combine_heads(attn_output)  # (batch, seq_len, d_model)
        
        # 5. Final linear projection
        output = combined @ self.W_o
        
        return output


# ============================================================================
# KOMPONEN 5: FEED-FORWARD NETWORK
# ============================================================================

class FeedForward:
    """
    Position-wise Feed-Forward Network
    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
    """
    def __init__(self, d_model, d_ff):
        """
        Args:
            d_model: dimensi input/output
            d_ff: dimensi hidden layer (biasanya 4 * d_model)
        """
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Layer 1: d_model -> d_ff
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        
        # Layer 2: d_ff -> d_model
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        """
        Args:
            x: input, shape (batch, seq_len, d_model)
        
        Returns:
            output: shape (batch, seq_len, d_model)
        """
        # Layer 1 + GELU activation
        hidden = gelu(x @ self.W1 + self.b1)
        
        # Layer 2
        output = hidden @ self.W2 + self.b2
        
        return output


# ============================================================================
# KOMPONEN 6: LAYER NORMALIZATION
# ============================================================================

class LayerNormalization:
    """
    Layer Normalization: normalizes across the feature dimension
    """
    def __init__(self, d_model, epsilon=1e-6):
        """
        Args:
            d_model: dimensi features
            epsilon: small constant untuk stabilitas numerik
        """
        self.epsilon = epsilon
        
        # Learnable parameters
        self.gamma = np.ones(d_model)  # scale
        self.beta = np.zeros(d_model)  # shift
    
    def forward(self, x):
        """
        Args:
            x: input, shape (batch, seq_len, d_model)
        
        Returns:
            normalized output, shape sama dengan input
        """
        # Hitung mean dan variance across feature dimension (axis=-1)
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(variance + self.epsilon)
        
        # Scale and shift
        output = self.gamma * x_norm + self.beta
        
        return output


# ============================================================================
# KOMPONEN 7: DECODER BLOCK
# ============================================================================

class DecoderBlock:
    """
    Satu block dari Transformer Decoder
    Struktur: LayerNorm -> Multi-Head Attention -> Add -> LayerNorm -> FFN -> Add
    """
    def __init__(self, d_model, num_heads, d_ff):
        """
        Args:
            d_model: dimensi model
            num_heads: jumlah attention heads
            d_ff: dimensi feed-forward network
        """
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
    
    def forward(self, x, mask):
        """
        Args:
            x: input, shape (batch, seq_len, d_model)
            mask: causal mask
        
        Returns:
            output: shape (batch, seq_len, d_model)
        """
        # Multi-Head Attention block dengan pre-norm
        norm_x = self.norm1.forward(x)
        attn_output = self.attention.forward(norm_x, mask)
        x = x + attn_output  # Residual connection
        
        # Feed-Forward block dengan pre-norm
        norm_x = self.norm2.forward(x)
        ffn_output = self.ffn.forward(norm_x)
        x = x + ffn_output  # Residual connection
        
        return x


# ============================================================================
# KOMPONEN 8: FULL GPT DECODER MODEL
# ============================================================================

class GPTDecoder:
    """
    Full GPT-style Decoder Transformer
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len=512):
        """
        Args:
            vocab_size: ukuran vocabulary
            d_model: dimensi model
            num_heads: jumlah attention heads per block
            num_layers: jumlah decoder blocks
            d_ff: dimensi feed-forward network
            max_seq_len: panjang maksimal sequence
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token Embedding
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Stack of Decoder Blocks
        self.layers = [
            DecoderBlock(d_model, num_heads, d_ff) 
            for _ in range(num_layers)
        ]
        
        # Final Layer Normalization
        self.final_norm = LayerNormalization(d_model)
        
        # Output projection ke vocabulary size
        self.output_projection = np.random.randn(d_model, vocab_size) * 0.01
    
    def forward(self, token_ids):
        """
        Forward pass lengkap
        
        Args:
            token_ids: input token IDs, shape (batch_size, seq_len)
        
        Returns:
            logits: output logits, shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        
        # 1. Token Embedding
        x = self.token_embedding.forward(token_ids)  # (batch, seq_len, d_model)
        
        # 2. Positional Encoding
        x = self.pos_encoding.forward(x)
        
        # 3. Create Causal Mask
        mask = create_causal_mask(seq_len)
        
        # 4. Pass through Decoder Blocks
        for layer in self.layers:
            x = layer.forward(x, mask)
        
        # 5. Final Layer Normalization
        x = self.final_norm.forward(x)
        
        # 6. Project ke vocabulary size
        logits = x @ self.output_projection  # (batch, seq_len, vocab_size)
        
        return logits
    
    def predict_next_token(self, token_ids):
        """
        Prediksi token berikutnya dengan probabilitas
        
        Args:
            token_ids: input tokens, shape (batch_size, seq_len)
        
        Returns:
            probs: probability distribution untuk token berikutnya, shape (batch_size, vocab_size)
            predicted_tokens: token dengan probabilitas tertinggi, shape (batch_size,)
        """
        # Forward pass
        logits = self.forward(token_ids)  # (batch, seq_len, vocab_size)
        
        # Ambil logits untuk token terakhir
        last_logits = logits[:, -1, :]  # (batch, vocab_size)
        
        # Softmax untuk mendapatkan probabilitas
        probs = softmax(last_logits, axis=-1)
        
        # Prediksi token dengan probabilitas tertinggi
        predicted_tokens = np.argmax(probs, axis=-1)
        
        return probs, predicted_tokens


# ============================================================================
# TESTING & VALIDATION
# ============================================================================

def test_components():
    """Test individual components"""
    print("="*70)
    print("TESTING INDIVIDUAL COMPONENTS")
    
    # Test 1: Softmax
    print("\n[TEST 1] Softmax")
    x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    result = softmax(x)
    print(f"Input:\n{x}")
    print(f"Output:\n{result}")
    print(f"Sum (should be 1.0 for each row): {result.sum(axis=-1)}")
    print(f"  → Row 0 sum: {result.sum(axis=-1)[0]:.6f}")
    print(f"  → Row 1 sum: {result.sum(axis=-1)[1]:.6f}")
    assert np.allclose(result.sum(axis=-1), 1.0), "Softmax sum should be 1!"
    print("✅ PASSED")
    
    # Test 2: GELU
    print("\n[TEST 2] GELU Activation")
    x = np.array([-2, -1, 0, 1, 2])
    result = gelu(x)
    print(f"Input: {x}")
    print(f"Output: {result}")
    print("✅ PASSED")
    
    # Test 3: Causal Mask
    print("\n[TEST 3] Causal Mask")
    mask = create_causal_mask(5)
    print("Causal mask (5x5):")
    print(mask)
    print("✅ PASSED")
    
    # Test 4: Token Embedding
    print("\n[TEST 4] Token Embedding")
    vocab_size = 100
    d_model = 32
    token_emb = TokenEmbedding(vocab_size, d_model)
    tokens = np.array([[1, 2, 3], [4, 5, 6]])
    embedded = token_emb.forward(tokens)
    print(f"Input tokens shape: {tokens.shape}")
    print(f"Embedded shape: {embedded.shape}")
    assert embedded.shape == (2, 3, 32), "Embedding shape mismatch!"
    print("✅ PASSED")
    
    # Test 5: Positional Encoding
    print("\n[TEST 5] Positional Encoding")
    pos_enc = PositionalEncoding(d_model=32, max_seq_len=100)
    result = pos_enc.forward(embedded)
    print(f"After positional encoding shape: {result.shape}")
    assert result.shape == embedded.shape, "Positional encoding shape mismatch!"
    print("✅ PASSED")
    
    # Test 6: Scaled Dot-Product Attention
    print("\n[TEST 6] Scaled Dot-Product Attention")
    batch, seq_len, d_k = 2, 4, 8
    Q = np.random.randn(batch, seq_len, d_k)
    K = np.random.randn(batch, seq_len, d_k)
    V = np.random.randn(batch, seq_len, d_k)
    mask = create_causal_mask(seq_len)
    output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
    print(f"Q, K, V shape: {Q.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention weights sum (should be ~1.0): {attn_weights.sum(axis=-1)}")
    print(f"\nAttention weights (first sample):")
    print(attn_weights[0])
    print("Note: Upper triangular should be 0 (because of causal mask)")
    print("✅ PASSED")
    
    print("\n")
    print("ALL COMPONENT TESTS PASSED! ✅")
    print("="*70)


def test_full_model():
    """Test full GPT model"""
    print("TESTING FULL GPT MODEL")
    
    # Hyperparameters (small untuk testing)
    VOCAB_SIZE = 1000
    D_MODEL = 128
    NUM_HEADS = 4
    NUM_LAYERS = 2
    D_FF = 512
    MAX_SEQ_LEN = 100
    
    print(f"\nModel Configuration:")
    print(f"  Vocabulary Size: {VOCAB_SIZE}")
    print(f"  Model Dimension: {D_MODEL}")
    print(f"  Number of Heads: {NUM_HEADS}")
    print(f"  Number of Layers: {NUM_LAYERS}")
    print(f"  FFN Dimension: {D_FF}")
    print(f"  Max Sequence Length: {MAX_SEQ_LEN}")
    
    # Create model
    print(f"\n[1] Creating GPT model...")
    model = GPTDecoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN
    )
    print("✅ Model created successfully")
    
    # Create dummy input
    BATCH_SIZE = 2
    SEQ_LEN = 10
    print(f"\n[2] Creating dummy input...")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Sequence length: {SEQ_LEN}")
    
    dummy_tokens = np.random.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    print(f"  Input tokens shape: {dummy_tokens.shape}")
    print(f"  Sample tokens: {dummy_tokens[0]}")
    
    # Forward pass
    print(f"\n[3] Running forward pass...")
    logits = model.forward(dummy_tokens)
    print(f"✅ Forward pass successful")
    print(f"  Output logits shape: {logits.shape}")
    
    # Validate shape
    expected_shape = (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    assert logits.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {logits.shape}"
    print(f"✅ Output shape is correct: {logits.shape}")
    
    # Test next token prediction
    print(f"\n[4] Testing next token prediction...")
    probs, predicted = model.predict_next_token(dummy_tokens)
    print(f"  Probability distribution shape: {probs.shape}")
    print(f"  Predicted next tokens: {predicted}")
    print(f"  Probability sum (should be ~1.0): {probs.sum(axis=-1)}")
    
    assert np.allclose(probs.sum(axis=-1), 1.0), "Probabilities should sum to 1!"
    print("✅ Probability distribution is valid")
    
    # Show top-k predictions
    print(f"\n[5] Top-5 predictions for first sample:")
    top_k = 5
    top_indices = np.argsort(probs[0])[-top_k:][::-1]
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. Token {idx}: {probs[0, idx]:.4f}")
    
    print("\n")
    print("FULL MODEL TEST PASSED! 🎉")
    print("="*70)
    
    return model, dummy_tokens, logits


def visualize_attention_pattern():
    """Visualize attention patterns (bonus feature)"""

    print("BONUS: VISUALIZING ATTENTION PATTERNS")
    
    # Create small example
    seq_len = 8
    d_k = 16
    
    Q = np.random.randn(1, seq_len, d_k)
    K = np.random.randn(1, seq_len, d_k)
    V = np.random.randn(1, seq_len, d_k)
    mask = create_causal_mask(seq_len)
    
    _, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(attn_weights[0], cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Causal Self-Attention Pattern')
    
    # Add grid
    plt.xticks(range(seq_len))
    plt.yticks(range(seq_len))
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('attention_pattern.png', dpi=150, bbox_inches='tight')
    print("✅ Attention pattern saved to 'attention_pattern.png'")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" TRANSFORMER FROM SCRATCH - IMPLEMENTATION TEST")
    
    # Test individual components
    test_components()
    
    # Test full model
    model, tokens, logits = test_full_model()
    
    # Bonus: Visualize attention
    try:
        visualize_attention_pattern()
    except Exception as e:
        print(f"\nNote: Visualization skipped ({e})")
    
    print("\n")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("="*70)