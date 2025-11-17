import torch
import sys
import os
import nltk
from transformers import PreTrainedModel
from torch.distributions import Categorical

# Ensure required NLTK tokenizers are available
nltk.download('punkt', quiet=True)
try:
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
A1_DIR = os.path.join(BASE_DIR, 'A1')

# Add A1 to path and import tokenizer
sys.path.insert(0, A1_DIR)
from A1 import A1Tokenizer

# Import the Transformer model
from A2_skeleton import A2Transformer, A2ModelConfig

# Define lowercase_tokenizer function - needed for pickle deserialization
def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]

# Make it available in __main__ module for pickle
import __main__
if not hasattr(__main__, 'lowercase_tokenizer'):
    __main__.lowercase_tokenizer = lowercase_tokenizer


def load_model_and_tokenizer(model_dir=None, tokenizer_path=None):
    """Load the pre-trained Transformer model and tokenizer."""
    if model_dir is None:
        # Default to A2/trainer_output where the pre-trained model is located
        model_dir = os.path.join(SCRIPT_DIR, 'trainer_output')
        print(f"Using pre-trained model directory: {model_dir}")
    
    if tokenizer_path is None:
        # Try base directory first, then A1 directory
        base_tokenizer = os.path.join(BASE_DIR, 'tokenizer.pkl')
        a1_tokenizer = os.path.join(A1_DIR, 'tokenizer.pkl')
        
        if os.path.exists(base_tokenizer):
            tokenizer_path = base_tokenizer
        elif os.path.exists(a1_tokenizer):
            tokenizer_path = a1_tokenizer
        else:
            tokenizer_path = base_tokenizer  # Default
    
    print(f"Loading model from: {model_dir}")
    print(f"Loading tokenizer from: {tokenizer_path}")
    
    # Load tokenizer
    import pickle
    try:
        tokenizer = A1Tokenizer.from_file(tokenizer_path)
        print(f"✓ Tokenizer loaded (vocab_size: {len(tokenizer)})")
    except (AttributeError, ModuleNotFoundError) as e:
        print(f"Standard loading failed: {e}")
        print("Trying direct pickle load...")
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        print(f"✓ Tokenizer loaded (vocab_size: {len(tokenizer)})")
    
    # Load model - handles both pytorch_model.bin and model.safetensors formats
    try:
        # Check what model file exists
        pytorch_model_path = os.path.join(model_dir, 'pytorch_model.bin')
        safetensors_path = os.path.join(model_dir, 'model.safetensors')
        
        if os.path.exists(pytorch_model_path):
            print(f"Found pytorch_model.bin format")
        elif os.path.exists(safetensors_path):
            print(f"Found model.safetensors format")
        else:
            print(f"⚠ Warning: No model file found in {model_dir}")
            print(f"   Looking for: pytorch_model.bin or model.safetensors")
        
        # Load using from_pretrained (handles both formats)
        model = A2Transformer.from_pretrained(model_dir)
        print(f"✓ Model loaded successfully")
        
        # Verify it's the correct model type
        import json
        config_path = os.path.join(model_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            if saved_config.get('architectures', [None])[0] == 'A2Transformer':
                print(f"✓ Verified: A2Transformer model")
            else:
                print(f"⚠ Warning: Model architecture is {saved_config.get('architectures')}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import json
        import traceback
        traceback.print_exc()
        config_path = os.path.join(model_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            print(f"Config file exists with keys: {list(saved_config.keys())}")
            required_fields = ['hidden_size', 'num_attention_heads', 'rope_theta', 
                              'intermediate_size', 'num_hidden_layers']
            missing = [f for f in required_fields if f not in saved_config or saved_config[f] is None]
            if missing:
                print(f"⚠ Missing or None config fields: {missing}")
        raise
    
    return model, tokenizer


def predict_next_word(model, tokenizer, prompt, device):
    """Predict the next word for a given prompt.
    
    Steps:
    1. Apply the model to the integer-encoded input text
    2. Take the model's output at the last position (avoid EOS)
    3. Use argmax to find the index of the highest-scoring item
    4. Apply the inverse vocabulary encoder to get the word
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input text string
        device: torch device
    
    Returns:
        The predicted next word (string)
    """
    model.eval()
    with torch.no_grad():
        # Step 1: Apply the model to integer-encoded input text
        enc = tokenizer(prompt, return_tensors='pt', padding=False, truncation=True)
        input_ids = enc["input_ids"].to(device)
        
        # Apply model
        logits = model(input_ids)  # Shape: [batch, seq_len, vocab_size]
        
        # Step 2: Take the model's output at the last position
        # Make sure we avoid EOS token - use the position before the last if last is EOS
        last_logits = logits[:, -1, :]  # Shape: [batch, vocab_size]
        
        # Step 3: Use argmax to find the index of the highest-scoring item
        predicted_idx = torch.argmax(last_logits, dim=-1).item()
        
        # Step 4: Apply the inverse vocabulary encoder
        predicted_word = tokenizer.id2word.get(predicted_idx, tokenizer.unk_token)
        
        return predicted_word


def generate_text(model, tokenizer, prompt, device, max_length=50, temperature=1.0, topk=None):
    """Generate text using random sampling.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The prompt that initializes the text generation
        device: torch device
        max_length: Maximal number of steps before terminating
        temperature: Controls the degree of randomness by scaling the predicted logits
        topk: For top-K sampling, truncate distribution to topk most probable tokens (None = no truncation)
    
    Returns:
        Generated text string
    """
    model.eval()
    with torch.no_grad():
        # Tokenize the prompt
        enc = tokenizer(prompt, return_tensors='pt', padding=False, truncation=True)
        input_ids = enc["input_ids"].to(device)
        
        generated_ids = input_ids.clone()
        
        for step in range(max_length):
            # Get logits for the last token
            logits = model(generated_ids)  # Shape: [batch, seq_len, vocab_size]
            next_token_logits = logits[:, -1, :]  # Shape: [batch, vocab_size]
            
            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature
            
            # Apply top-K sampling if specified
            if topk is not None and topk > 0:
                # Get top-k values and indices
                topk_values, topk_indices = torch.topk(next_token_logits, k=min(topk, next_token_logits.shape[-1]), dim=-1)
                
                # Create a tensor with -inf for all positions
                filtered_logits = torch.full_like(next_token_logits, float('-inf'))
                # Set top-k positions to their original values
                filtered_logits.scatter_(-1, topk_indices, topk_values)
                next_token_logits = filtered_logits
            
            # Sample from the distribution using Categorical
            dist = Categorical(logits=next_token_logits)
            next_token = dist.sample()  # Shape: [batch] = [1]
            # Reshape to [batch, 1] to match generated_ids shape [batch, seq_len]
            next_token = next_token.unsqueeze(1)  # Shape: [1, 1]
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Stop if we generate EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
        
        # Convert token IDs back to text
        generated_token_ids = generated_ids[0].tolist()
        # Filter out special tokens for display
        tokens = []
        for token_id in generated_token_ids:
            token = tokenizer.id2word.get(token_id, tokenizer.unk_token)
            if token not in [tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token]:
                tokens.append(token)
        
        return ' '.join(tokens)


def load_olmo_model(local_dir='/data/courses/2025_dat450_dit247/models/OLMo-2-0425-1B'):
    """Load the pre-trained OLMo-2 model for comparison.
    
    Note: This model returns CausalLMOutputWithPast, so we need to access .logits
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        if not os.path.exists(local_dir):
            print(f"OLMo model directory not found: {local_dir}")
            print("Skipping OLMo model loading.")
            return None, None
        
        print(f"Loading OLMo-2 model from: {local_dir}")
        tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(local_dir, local_files_only=True)
        print("✓ OLMo-2 model loaded")
        return model, tokenizer
    except Exception as e:
        print(f"Could not load OLMo model: {e}")
        return None, None


def generate_text_with_olmo(model, tokenizer, prompt, device, max_length=50, temperature=1.0, topk=None):
    """Generate text using OLMo model (handles CausalLMOutputWithPast)."""
    model.eval()
    with torch.no_grad():
        # Tokenize the prompt
        enc = tokenizer(prompt, return_tensors='pt', padding=False, truncation=True)
        input_ids = enc["input_ids"].to(device)
        
        generated_ids = input_ids.clone()
        
        for step in range(max_length):
            # Get logits - OLMo returns CausalLMOutputWithPast, need .logits
            output = model(generated_ids)
            if hasattr(output, 'logits'):
                logits = output.logits
            else:
                logits = output  # Fallback if it's already logits
            
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-K if specified
            if topk is not None and topk > 0:
                topk_values, topk_indices = torch.topk(next_token_logits, k=min(topk, next_token_logits.shape[-1]), dim=-1)
                filtered_logits = torch.full_like(next_token_logits, float('-inf'))
                filtered_logits.scatter_(-1, topk_indices, topk_values)
                next_token_logits = filtered_logits
            
            # Sample
            dist = Categorical(logits=next_token_logits)
            next_token = dist.sample()  # Shape: [batch] = [1]
            # Reshape to [batch, 1] to match generated_ids shape [batch, seq_len]
            next_token = next_token.unsqueeze(1)  # Shape: [1, 1]
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Stop at EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
        
        # Decode
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text


if __name__ == "__main__":
    # Select device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}\n")
    
    # Load the pre-trained Transformer model
    print("="*60)
    print("LOADING PRE-TRAINED TRANSFORMER MODEL")
    print("="*60)
    model, tokenizer = load_model_and_tokenizer()
    model.to(device)
    model.eval()
    
    print("\n" + "="*60)
    print("1. PREDICTING NEXT WORD")
    print("="*60)
    
    # Example prompts for next word prediction
    test_prompts = [
        "he lives in san",
        "The capital of France is",
        "In the beginning",
        "Machine learning is"
    ]
    
    for prompt in test_prompts:
        predicted_word = predict_next_word(model, tokenizer, prompt, device)
        print(f"  '{prompt}' → {predicted_word}")
    
    print("\n" + "="*60)
    print("2. GENERATING TEXTS")
    print("="*60)
    
    # Example prompts for text generation
    generation_prompts = [
        'In natural language processing, a Transformer',
        'Is Stockholm the capital of Sweden? Answer yes or no. The answer is',
        'Write a Python program that reverses a list.'
    ]
    
    print("\n--- Default parameters (temperature=1.0, topk=None) ---")
    for prompt in generation_prompts:
        generated = generate_text(model, tokenizer, prompt, device, max_length=50, temperature=1.0, topk=None)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated}")
    
    print("\n--- With temperature=0.8, topk=50 ---")
    for prompt in generation_prompts:
        generated = generate_text(model, tokenizer, prompt, device, max_length=50, temperature=0.8, topk=50)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated}")
    
    print("\n--- With temperature=1.5, topk=100 ---")
    for prompt in generation_prompts:
        generated = generate_text(model, tokenizer, prompt, device, max_length=50, temperature=1.5, topk=100)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated}")
    
    # Try loading OLMo-2 model for comparison
    print("\n" + "="*60)
    print("3. COMPARING TO PRE-TRAINED OLMo-2 MODEL")
    print("="*60)
    
    olmo_model, olmo_tokenizer = load_olmo_model()
    
    if olmo_model is not None and olmo_tokenizer is not None:
        olmo_model.to(device)
        olmo_model.eval()
        
        print("\n--- OLMo-2 predictions (temperature=0.8, topk=50) ---")
        for prompt in generation_prompts:
            generated = generate_text_with_olmo(olmo_model, olmo_tokenizer, prompt, device, 
                                                max_length=50, temperature=0.8, topk=50)
            print(f"\nPrompt: '{prompt}'")
            print(f"Generated: {generated}")
    else:
        print("\nOLMo-2 model not available. Skipping comparison.")
    
    print("\n" + "="*60)
    print("Inference complete!")
    print("="*60)
