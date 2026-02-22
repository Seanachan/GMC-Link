"""
GMC-Link Training Script
========================
Trains the MotionLanguageAligner using BCE loss with motion-language pairs
from refer-kitti sequences 0015/0016/0018 (test on 0011).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from gmc_link.losses import AlignmentLoss
from gmc_link.alignment import MotionLanguageAligner
from gmc_link.dataset import MotionLanguageDataset, collate_fn, build_training_data
from gmc_link.text_utils import TextEncoder

def train_one_epoch(model, dataloader, optimizer, loss_func, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (motion_features, language_features, labels) in enumerate(dataloader):
        motion_features = motion_features.to(device)
        language_features = language_features.to(device)
        labels = labels.to(device)
        
        # Per-pair similarity scores
        scores = model.score_pairs(motion_features, language_features)
        
        loss = loss_func(scores, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Track accuracy
        preds = (torch.sigmoid(scores) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    accuracy = correct / total if total > 0 else 0
    return total_loss / len(dataloader), accuracy

def main():
    # --- 1. Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    learning_rate = 1e-3
    batch_size = 128
    epochs = 200
    
    # Refer-KITTI data paths (Train on 15/16/18, test on 11)
    data_root = "refer-kitti"
    sequences = ["0015", "0016", "0018"]
    
    # --- 2. Initialize Text Encoder ---
    print("Loading text encoder...")
    encoder = TextEncoder(device=str(device))
    lang_dim = 384

    # --- 3. Load Data ---
    print("Building training data...")
    all_motions, all_languages, all_labels = build_training_data(
        data_root=data_root,
        sequences=sequences,
        text_encoder=encoder,
        frame_gap=5,
    )
    
    print(f"Total training samples: {len(all_motions)}")
    
    if len(all_motions) == 0:
        print("ERROR: No training data found.")
        return

    # --- 4. Initialize Model ---
    model = MotionLanguageAligner(
        motion_dim=2, 
        lang_dim=lang_dim, 
        embed_dim=256
    ).to(device)
    
    criterion = AlignmentLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # --- 5. Create DataLoader ---
    dataset = MotionLanguageDataset(all_motions, all_languages, all_labels)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # --- 6. Training Loop ---
    print(f"Starting training on {device} | {len(dataloader)} batches/epoch...")
    for epoch in tqdm(range(epochs)):
      avg_loss, accuracy = train_one_epoch(model, dataloader, optimizer, criterion, device)
      scheduler.step()
      
      if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Acc: {accuracy:.2%} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # --- 7. Save ---
    torch.save(model.state_dict(), "gmc_link_weights.pth")
    print("Training complete. Weights saved to gmc_link_weights.pth")

if __name__ == "__main__":
    main()