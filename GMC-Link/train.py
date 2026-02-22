import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .alignment import MotionLanguageAligner
from .losses import AlignmentContrastiveLoss
from .dataset import MotionDirectionDataset, collate_fn

def train_one_epoch(model, dataloader, optimizer, loss_func, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (motion_features, language_features) in enumerate(dataloader):
        # Move data to GPU/CPU
        motion_features = motion_features.to(device)
        language_features = language_features.to(device)
        
        # 1. Forward Pass: Get the N x N alignment matrix
        alignment_logits = model(motion_features, language_features)
        
        # 2. Calculate Loss: Compare diagonal to off-diagonal
        loss = loss_func(alignment_logits)
        
        # 3. Backpropagation: Update the "Brain"
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def main():
    # --- 1. Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 1e-4
    batch_size = 16 
    epochs = 50
    
    # --- 2. Initialize GMC-Link Components ---
    model = MotionLanguageAligner(
        motion_input_dim=2, 
        lang_input_dim=768, 
        shared_embed_dim=256
    ).to(device)
    
    criterion = AlignmentContrastiveLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # --- 3. Mock Data for Direction/State ---
    # In a real scenario, you'd load this from Refer-KITTI
    # Here, we create pairs: [1, 0] -> "Right", [-1, 0] -> "Left", [0, 0] -> "Stationary"
    mock_motions = [
        np.array([1.0, 0.0]),   # Right
        np.array([-1.0, 0.0]),  # Left
        np.array([0.0, 1.0]),   # Down
        np.array([0.0, 0.0])    # Stationary
    ] * 100 # Create 400 samples
    
    # Mock embeddings (random for now, replace with CLIP/BERT outputs)
    mock_texts = [torch.randn(768) for _ in range(400)]
    
    dataset = MotionDirectionDataset(mock_motions, mock_texts)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    # --- 4. Training Loop ---
    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

    # --- 5. Save the Weights ---
    torch.save(model.state_dict(), "gmc_link_weights.pth")
    print("Training complete. Weights saved to gmc_link_weights.pth")

if __name__ == "__main__":
    import numpy as np # Needed for mock data
    main()