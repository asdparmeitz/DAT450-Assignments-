import torch
import sys
import os
from torch.utils.data import DataLoader, Subset
import numpy as np
import pickle
import nltk
from datasets import load_dataset

nltk.download('punkt', quiet=True)
try:
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

from A2_skeleton import A2Transformer, A2ModelConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
A1_DIR = os.path.join(BASE_DIR, 'A1')
train_path = os.path.join(A1_DIR, "train.txt")
val_path = os.path.join(A1_DIR, "val.txt")

def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]

import __main__
if not hasattr(__main__, 'lowercase_tokenizer'):
    __main__.lowercase_tokenizer = lowercase_tokenizer

sys.path.insert(0, A1_DIR)
print("Importing A1Tokenizer...", flush=True)
from A1 import A1Tokenizer
print("A1Tokenizer imported successfully", flush=True)

class TrainingArguments:
    def __init__(self):
        self.optim = 'adamw_torch'
        self.eval_strategy = 'epoch'
        self.use_cpu = False
        self.no_cuda = False
        self.learning_rate = 1e-4
        self.num_train_epochs = 5
        self.per_device_train_batch_size = 64
        self.per_device_eval_batch_size = 64
        self.output_dir = os.path.join(SCRIPT_DIR, 'trainer_output')

class A1Trainer:

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        assert(args.optim == 'adamw_torch')
        assert(args.eval_strategy == 'epoch')

    def select_device(self):
        if self.args.use_cpu:
            return torch.device('cpu')
        if not self.args.no_cuda and torch.cuda.is_available():
            return torch.device('cuda')
        if torch.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
            
    def train(self):
        args = self.args

        device = self.select_device()
        print('Device:', device, flush=True)
        
        if device.type == 'cuda':
            print(f'GPU:{torch.cuda.get_device_name(0)}', flush=True)

        else:
            print('Using cPU', flush=True)
        
        self.model.to(device)
        
        V = self.model.config.vocab_size
        
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        optimizer = torch.optim.AdamW(
                        self.model.parameters(),
                        lr = args.learning_rate,
                        weight_decay=0.01,
                    )

        train_loader = DataLoader(self.train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
        val_loader = DataLoader(self.eval_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False)

        for epoch in range(args.num_train_epochs):
            self.model.train()
            train_loss_sum, train_tok = 0.0, 0
            
            num_batches = len(train_loader)
            print(f'\nEpoch {epoch+1}/{args.num_train_epochs} - {num_batches} batches', flush=True)

            for batch_idx, batch in enumerate(train_loader):
                texts = batch["text"]
                enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)

                X = input_ids[:, :-1]
                Y = input_ids[:, 1:]
                
                if batch_idx == 0:
                    max_token_id_Y = Y.max().item()
                    min_token_id_Y = Y.min().item()
                    max_token_id_X = X.max().item()
                    min_token_id_X = X.min().item()
                    print(f"DEBUG: X: {min_token_id_X}, max: {max_token_id_X}, vocab_size: {V}", flush=True)
                    print(f"DEBUG: Y {min_token_id_Y}, max: {max_token_id_Y}, vocab_size: {V}", flush=True)
                
                max_token_id_X = X.max().item()
                min_token_id_X = X.min().item()
                max_token_id_Y = Y.max().item()
                min_token_id_Y = Y.min().item()
                if max_token_id_X >= V or min_token_id_X < 0 or max_token_id_Y >= V or min_token_id_Y < 0:
                    if batch_idx == 0:
                        print("AAAAAAA")
                    X = torch.clamp(X, 0, V - 1)
                    Y = torch.clamp(Y, 0, V - 1)

                try:
                    torch.cuda.synchronize()
                    logits = self.model(X)
                    torch.cuda.synchronize()
                    loss = loss_func(logits.reshape(-1, V), Y.reshape(-1))
                    torch.cuda.synchronize()
                except RuntimeError as e:
                    if "CUDA" in str(e) or "device-side assert" in str(e):
                        print(f"CUDA Error {batch_idx}", flush=True)
                        print(f"X shape: {X.shape}, Y shape: {Y.shape}", flush=True)
                        X_cpu = X.cpu()
                        Y_cpu = Y.cpu()
                        print(f"X min/max {X_cpu.min().item()}/{X_cpu.max().item()}, Y min/max {Y_cpu.min().item()}/{Y_cpu.max().item()}", flush=True)
                        print(f"X sample {X_cpu[0, :20].tolist()}", flush=True)
                        print(f"Y sample {Y_cpu[0, :20].tolist()}", flush=True)
                        raise
                    else:
                        raise

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.cuda.synchronize()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                torch.cuda.synchronize()

                with torch.no_grad():
                    Y_cpu = Y.cpu()
                    num_tokens = (Y_cpu != self.tokenizer.pad_token_id).sum().item()
                    train_loss_sum += loss.item() * num_tokens
                    train_tok += num_tokens
                
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                    current_loss = train_loss_sum / max(1, train_tok)
                    print(f'  Batch {batch_idx+1}/{num_batches}: Loss={current_loss:.4f}, Tokens={train_tok}', flush=True)

            self.model.eval()
            with torch.no_grad():
                val_loss_sum, val_tok = 0.0, 0
                for batch in val_loader:
                    texts = batch["text"]
                    enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                    ids = enc["input_ids"].to(device)
                    Xv, Yv = ids[:, :-1], ids[:, 1:]
                    
                    if Yv.max().item() >= V or Yv.min().item() < 0:
                        Yv = torch.clamp(Yv, 0, V - 1)
                        Xv = torch.clamp(Xv, 0, V - 1)
                    torch.cuda.synchronize()
                    logits = self.model(Xv)
                    torch.cuda.synchronize()
                    loss_v = loss_func(logits.reshape(-1, V), Yv.reshape(-1))
                    torch.cuda.synchronize()
                    Yv_cpu = Yv.cpu()
                    num_tokens = (Yv_cpu != self.tokenizer.pad_token_id).sum().item()
                    val_loss_sum += loss_v.item() * num_tokens
                    val_tok += num_tokens

            train_ce = train_loss_sum / max(1, train_tok)
            val_ce = val_loss_sum / max(1, val_tok)
            train_ppl = float(np.exp(train_ce))
            val_ppl = float(np.exp(val_ce))

            print(f"Epoch {epoch+1}: train CE={train_ce:.4f} PPL={train_ppl:.1f} | val CE={val_ce:.4f} PPL={val_ppl:.1f}", flush=True)
            self.model.train()

        print(f'Saving to {args.output_dir}.', flush=True)
        self.model.save_pretrained(args.output_dir, safe_serialization=False)


dataset = load_dataset('text', data_files={'train': train_path, 'val': val_path})
dataset = dataset.filter(lambda x: x['text'].strip() != '')

TEST_MODE = False
if TEST_MODE:
    print("TEST MODE: Using reduced dataset size (100 train, 50 val samples)", flush=True)
    train_dataset = Subset(dataset["train"], range(100))
    eval_dataset = Subset(dataset["val"], range(50))
else:
    train_dataset = dataset["train"]
    eval_dataset = dataset["val"]

try:
    tokenizer = A1Tokenizer.from_file(os.path.join(A1_DIR, 'tokenizer.pkl'))
except (AttributeError, ModuleNotFoundError):
    with open(os.path.join(A1_DIR, 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)
print(f"Loaded tokenizer with vocab_size: {len(tokenizer)}", flush=True)

config = A2ModelConfig(
    vocab_size=len(tokenizer),
    hidden_size=256,
    intermediate_size=1024,
    num_attention_heads=4,
    num_hidden_layers=2,
    rope_theta=10000.0,
    hidden_act='silu',
    max_position_embeddings=256,
    rms_norm_eps=1e-6
)


model = A2Transformer(config)
print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)

args = TrainingArguments()
print(f"\nModel will be saved to: {args.output_dir}", flush=True)


trainer = A1Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)



print("STARTING TRANSFORMER TRAINING", flush=True)

trainer.train()


print("TRANSFORMER TRAINING COMPLETED!", flush=True)


saved_config_path = os.path.join(args.output_dir, 'config.json')
if os.path.exists(saved_config_path):
    import json
    with open(saved_config_path, 'r') as f:
        saved_config = json.load(f)
    if saved_config.get('architectures', [None])[0] == 'A2Transformer':
        print(f"Verified: Saved model is A2Transformer in {args.output_dir}", flush=True)
    else:
        print(f"WARNING: Saved model architecture is {saved_config.get('architectures')}, expected A2Transformer!", flush=True)
else:
    print(f"WARNING: Could not find config.json at {saved_config_path}", flush=True)

print(f"Saving model to {args.output_dir}...", flush=True)
model.save_pretrained(args.output_dir, safe_serialization=False)
print(f"Model saved to {args.output_dir}", flush=True)
