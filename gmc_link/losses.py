import torch
import torch.nn as nn

class AlignmentLoss(nn.Module):
    """
    Computes the alignment loss between motion features and language features.
    This encourages the model to learn a shared representation where related motion and language features are close together.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.loss_function = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, alignment_logits):
        """
        Args:
            alignment_logits: (N, N) Tensor of similarity scores between motion and language features, where N is the number of tracks.
        Returns:
            loss: Scalar Tensor - The computed alignment loss. Scalar tensor for backpropagation.
        """

        # Number of samples in the batch (number of tracks)
        batch_size = alignment_logits.size(0)

        # 1. Create Ground Truth Labels
        # we always arrange it so that Motion #0 matches Text #0. This means the correct answers are always located on the diagonal of the matrix. These indices tell the loss function exactly where the "True" matches are hiding.I
        target_indices = torch.arange(batch_size, device=alignment_logits.device)

        # 2. Symmetric Loss (CLIP Style)
        # We check two things:
        # A. For each Motion, which Text is the best match? (Row-wise)
        loss_motion_to_text = self.loss_function(alignment_logits, target_indices)

        # B. For each Text, which Motion is the best match? (Column-wise)
        # We transpose the logits to look at it from the text's perspective.
        loss_text_to_motion = self.loss_function(alignment_logits.t(), target_indices)

        # 3. Average the two perspectives
        # This 'Symmetric' approach makes the alignment much more stable.
        total_loss = (loss_motion_to_text + loss_text_to_motion) / 2

        return total_loss
