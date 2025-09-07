# Import necessary libraries
import os
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import cv2

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ====================== DATA HANDLING ======================

class SyntheticDigitSequenceDataset(Dataset):
    """
    A dataset class that generates synthetic sequences of handwritten digits
    by combining multiple MNIST digits into a single image.
    """

    def __init__(self, base_dataset, min_length=1, max_length=5, img_width=160, img_height=32, transform=None,
               train=True):
        self.base_dataset = base_dataset
        self.min_length = min_length
        self.max_length = max_length
        self.img_width = img_width
        self.img_height = img_height
        self.transform = transform
        self.train = train

        # Filter MNIST dataset to ensure we have enough digits
        self.digits_by_class = {i: [] for i in range(10)}
        for idx in range(len(base_dataset)):
            digit, label = base_dataset[idx]
            self.digits_by_class[label].append(digit)

        # Use global MNIST mean and std for consistency
        self.mean = 0.1307
        self.std = 0.3081

    def __len__(self):
        return 30000  # Reduced to 30,000 synthetic sequences for faster training

    def __getitem__(self, idx):
        # Randomly determine the length of the sequence
        seq_length = random.randint(self.min_length, self.max_length)
        canvas = Image.new('L', (self.img_width, self.img_height), color=0)

        # Randomly select digits for the sequence
        digits = []
        labels = []
        for _ in range(seq_length):
            digit_class = random.randint(0, 9)
            digit_img = random.choice(self.digits_by_class[digit_class])
            digits.append(digit_img)
            labels.append(str(digit_class))

        # Calculate positions for each digit with random spacing
        x_pos = random.randint(5, 15)  # Start with some margin
        digit_positions = []

        for digit in digits:
            # Convert tensor to PIL Image if needed
            if isinstance(digit, torch.Tensor):
                digit = transforms.ToPILImage()(digit)

            # Randomly resize digit (simulating different writing sizes)
            width_percent = random.uniform(0.7, 1.0)
            new_height = int(self.img_height * width_percent)
            digit = digit.resize((new_height, new_height), Image.Resampling.LANCZOS)

            # Calculate position with random spacing
            spacing = random.randint(3, 8)  # Increased spacing
            x_pos += spacing

            # Center digit vertically with some randomness
            y_pos = (self.img_height - new_height) // 2 + random.randint(-3, 3)

            # Paste digit onto canvas
            canvas.paste(digit, (x_pos, y_pos))

            # Update x position for next digit
            x_pos += new_height
            digit_positions.append((x_pos, y_pos))

        # Apply data augmentation if training
        if self.transform:
            canvas = self.transform(canvas)
        else:
            # Fallback (should not generally happen if we provide transforms)
            canvas = transforms.ToTensor()(canvas)
            canvas = transforms.Normalize((self.mean,), (self.std,))(canvas)

        # Convert labels to tensor
        label_str = ''.join(labels)
        label_tensor = torch.tensor([int(c) for c in label_str], dtype=torch.long)

        return canvas, label_tensor, label_str


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads images to the same width and labels to the same length.
    """
    images, labels, label_strs = zip(*batch)

    # Images are already tensors of shape (1, H, W)
    # Stack images along the batch dimension
    images = torch.stack(images, dim=0)

    # Get sequence lengths for labels
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)

    # Pad labels to the maximum length in the batch
    labels = pad_sequence(labels, batch_first=True, padding_value=10)  # Use 10 as padding value

    return images, labels, label_lengths, label_strs


# ====================== MODEL ARCHITECTURE ======================

class CNNBackbone(nn.Module):
    """
    CNN backbone for feature extraction from input images.
    """

    def __init__(self, input_height=32):
        super(CNNBackbone, self).__init__()
        self.input_height = input_height

        # Calculate the output height after convolutions
        conv_output_height = input_height // 2 // 2  # 2 maxpool layers with stride 2

        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Depthwise separable convolution for efficiency
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128),
            nn.Conv2d(128, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Adaptive pooling to handle variable width
        self.adaptive_pool = nn.AdaptiveAvgPool2d((conv_output_height, None))

    def forward(self, x):
        # x shape: (B, C, H, W) = (B, 1, 32, W)
        x = self.conv_layers(x)  # (B, 256, H/4, W/4)
        x = self.adaptive_pool(x)  # (B, 256, conv_output_height, W')

        # Rearrange dimensions for RNN: (B, C, H, W) -> (B, W, C * H)
        x = x.permute(0, 3, 1, 2)  # (B, W', 256, conv_output_height)
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.contiguous().view(batch_size, seq_len, -1)  # (B, W', 256 * conv_output_height)

        return x


class CRNN(nn.Module):
    """
    Combined CNN and RNN model for sequence recognition.
    """

    def __init__(self, input_height=32, num_classes=11, hidden_size=128, num_layers=2, dropout=0.3):
        super(CRNN, self).__init__()
        self.cnn = CNNBackbone(input_height)
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # Calculate the output feature size from CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_height, 100)  # Assume width=100
            dummy_output = self.cnn(dummy_input)
            rnn_input_size = dummy_output.size(2)

        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )

        # Output layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Extract features using CNN
        x = self.cnn(x)  # (B, W, C * H)
        x = self.dropout(x)

        # Apply RNN
        x, _ = self.rnn(x)  # (B, W, hidden_size * 2)
        x = self.dropout(x)

        # Apply fully connected layer
        x = self.fc(x)  # (B, W, num_classes)

        # Apply log softmax for CTC loss
        x = F.log_softmax(x, dim=2)

        return x


# ====================== TRAINING SETUP ======================

def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001,
                checkpoint_dir='checkpoints'):
    """
    Train the model with CTC loss and validation monitoring.
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.CTCLoss(blank=10, zero_infinity=True)  # Use 10 as blank token

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': []
    }

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels, label_lengths, label_strs) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            # Forward pass
            outputs = model(images)  # (B, W, C)
            outputs = outputs.permute(1, 0, 2)  # (W, B, C) for CTC loss

            # Calculate input lengths for CTC
            input_lengths = torch.full(
                size=(outputs.size(1),),
                fill_value=outputs.size(0),
                dtype=torch.long
            ).to(device)

            # Calculate loss
            loss = criterion(outputs, labels, input_lengths, label_lengths)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            # Calculate statistics
            train_loss += loss.item()

            # Decode predictions to calculate accuracy
            predictions = improved_decode_predictions(outputs.permute(1, 0, 2))  # Back to (B, W, C)
            for i, pred in enumerate(predictions):
                if pred == label_strs[i]:
                    train_correct += 1
            train_total += len(predictions)

            # Print progress
            if batch_idx % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {train_correct / train_total:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels, label_lengths, label_strs in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                label_lengths = label_lengths.to(device)

                # Forward pass
                outputs = model(images)
                outputs = outputs.permute(1, 0, 2)  # (W, B, C) for CTC loss

                # Calculate input lengths for CTC
                input_lengths = torch.full(
                    size=(outputs.size(1),),
                    fill_value=outputs.size(0),
                    dtype=torch.long
                ).to(device)

                # Calculate loss
                loss = criterion(outputs, labels, input_lengths, label_lengths)
                val_loss += loss.item()

                # Decode predictions to calculate accuracy
                predictions = improved_decode_predictions(outputs.permute(1, 0, 2))
                for i, pred in enumerate(predictions):
                    if pred == label_strs[i]:
                        val_correct += 1
                val_total += len(predictions)

        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # Print epoch results
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
              f'LR: {current_lr:.6f}')

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'val_loss': avg_val_loss,
            'val_acc': val_acc,
            'mean': 0.1307,
            'std': 0.3081
        }

        torch.save(checkpoint, os.path.join(checkpoint_dir, 'last_checkpoint.pth'))

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f'Best model saved with validation loss: {best_val_loss:.4f}')

        # Update learning rate
        scheduler.step(avg_val_loss)

    # Plot training history
    plot_training_history(history)

    return history


def plot_training_history(history):
    """Plot training and validation loss and accuracy."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_title('Loss Over Epochs')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['val_acc'], label='Validation Accuracy')
    axes[1].set_title('Accuracy Over Epochs')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # Plot learning rate
    axes[2].plot(history['lr'], label='Learning Rate')
    axes[2].set_title('Learning Rate Over Epochs')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Learning Rate')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()


# ====================== INFERENCE & EVALUATION ======================

def decode_predictions(predictions, blank_index=10):
    """
    Decode model predictions to string sequences using CTC greedy decoding.

    Args:
        predictions: Tensor of shape (B, W, C) with log probabilities
        blank_index: Index of the blank token

    Returns:
        List of decoded strings
    """
    # Get the most likely class at each time step
    _, max_indices = torch.max(predictions, dim=2)

    decoded_strings = []
    for batch in max_indices:
        # Convert to list and collapse repeated characters
        sequence = []
        previous_char = blank_index

        for char_idx in batch:
            char_idx = char_idx.item()
            if char_idx != blank_index and char_idx != previous_char:
                sequence.append(str(char_idx))
            previous_char = char_idx

        decoded_strings.append(''.join(sequence))

    return decoded_strings


def beam_search_decode(predictions, beam_width=10, blank_index=10):
    """
    Beam search decoding for CTC outputs
    """
    # predictions shape: (batch_size, seq_len, num_classes)
    batch_size, seq_len, num_classes = predictions.shape

    # Convert to probabilities (from log probs)
    probs = torch.exp(predictions)

    decoded_strings = []

    for b in range(batch_size):
        # Initialize beams: (sequence, probability)
        beams = [([], 1.0)]

        for t in range(seq_len):
            new_beams = []
            for beam in beams:
                sequence, score = beam
                for c in range(num_classes):
                    if c == blank_index:
                        # Blank doesn't change the sequence
                        new_beams.append((sequence, score * probs[b, t, c].item()))
                    else:
                        # Non-blank character
                        new_sequence = sequence + [c]
                        new_score = score * probs[b, t, c].item()
                        new_beams.append((new_sequence, new_score))

            # Keep only top-k beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        # Get best beam
        best_sequence, best_score = beams[0]

        # Convert to string, collapsing repeats and removing blanks
        decoded = []
        previous = blank_index
        for char_idx in best_sequence:
            if char_idx != blank_index and char_idx != previous:
                decoded.append(str(char_idx))
            previous = char_idx

        decoded_strings.append(''.join(decoded))

    return decoded_strings


def improved_decode_predictions(predictions, blank_index=10):
    """
    Improved CTC decoding with better handling of blank tokens and repetitions
    """
    # Get the most likely class at each time step
    _, max_indices = torch.max(predictions, dim=2)

    decoded_strings = []
    for batch in max_indices:
        # Convert to list
        sequence = batch.cpu().numpy().tolist()

        # CTC decoding: remove blanks and collapse repeats
        decoded = []
        previous = blank_index

        for char_idx in sequence:
            # Skip blank tokens
            if char_idx == blank_index:
                previous = blank_index
                continue

            # Only add characters that are different from previous
            if char_idx != previous:
                decoded.append(str(char_idx))
                previous = char_idx

        decoded_strings.append(''.join(decoded))

    return decoded_strings


def evaluate_model(model, test_loader):
    """
    Evaluate the model on the test set and return metrics.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, label_lengths, label_strs in test_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Decode predictions with improved decoding
            predictions = improved_decode_predictions(outputs)
            all_predictions.extend(predictions)
            all_labels.extend(label_strs)

            # Calculate accuracy
            for pred, true in zip(predictions, label_strs):
                if pred == true:
                    correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0

    # Calculate additional metrics
    try:
        from Levenshtein import distance as lev_distance

        lev_distances = []
        for pred, true in zip(all_predictions, all_labels):
            lev_distances.append(lev_distance(pred, true))

        avg_lev_distance = sum(lev_distances) / len(lev_distances) if lev_distances else 0
        perfect_match_ratio = sum(1 for d in lev_distances if d == 0) / len(lev_distances) if lev_distances else 0

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Average Levenshtein Distance: {avg_lev_distance:.4f}")
        print(f"Perfect Match Ratio: {perfect_match_ratio:.4f}")
    except ImportError:
        print(f"Accuracy: {accuracy:.4f}")
        avg_lev_distance = 0
        perfect_match_ratio = 0

    # Display some examples
    print("\nSample Predictions:")
    for i in range(min(10, len(all_predictions))):
        print(f"True: {all_labels[i]}, Pred: {all_predictions[i]}")

    return {
        'accuracy': accuracy,
        'avg_lev_distance': avg_lev_distance,
        'perfect_match_ratio': perfect_match_ratio,
        'predictions': all_predictions,
        'labels': all_labels
    }


# ====================== MAIN EXECUTION ======================
def main():
    """
    Main function to run the entire pipeline.
    """
    # Hyperparameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    MIN_SEQ_LENGTH = 1
    MAX_SEQ_LENGTH = 5
    IMG_HEIGHT = 32
    IMG_WIDTH = 160

    # Use standard MNIST mean and std for consistency
    GLOBAL_MEAN = 0.1307
    GLOBAL_STD = 0.3081

    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((GLOBAL_MEAN,), (GLOBAL_STD,)),
    ])

    # Test transform: convert to tensor and normalize (no randomness)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((GLOBAL_MEAN,), (GLOBAL_STD,))
    ])

    # Create base MNIST datasets
    mnist_train_base = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    mnist_test_base = MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    # Create synthetic datasets
    train_dataset = SyntheticDigitSequenceDataset(
        mnist_train_base, MIN_SEQ_LENGTH, MAX_SEQ_LENGTH, IMG_WIDTH, IMG_HEIGHT, transform=train_transform, train=True
    )

    test_dataset = SyntheticDigitSequenceDataset(
        mnist_test_base, MIN_SEQ_LENGTH, MAX_SEQ_LENGTH, IMG_WIDTH, IMG_HEIGHT, transform=test_transform, train=False
    )

    # Split train dataset into train and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # Initialize model
    model = CRNN(input_height=IMG_HEIGHT, num_classes=11, hidden_size=128, num_layers=2, dropout=0.3)
    model = model.to(device)

    # Check if we should train or load a pretrained model
    train_new_model = not os.path.exists('checkpoints/best_model.pth')

    if train_new_model:
        print("Training new model...")
        history = train_model(
            model, train_loader, val_loader,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            checkpoint_dir='checkpoints'
        )
    else:
        print("Loading pretrained model...")
        checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        history = checkpoint['history']

    # Evaluate the model
    print("Evaluating model on test set...")
    metrics = evaluate_model(model, test_loader)

    return model, metrics


def test_decoding():
    """Test the improved decoding function"""
    # Create a simple tensor that simulates the model output
    # Simulate a case where the model might add extra '3's
    test_tensor = torch.tensor([[
        [0.1, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # 3
        [0.1, 0.1, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # 4
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1],  # 5
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.1],  # 6
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7],  # blank (index 10)
    ]]).log()  # Convert to log probabilities

    predictions = improved_decode_predictions(test_tensor)
    print(f"Test decoding: {predictions}")  # Should be "3456" (not "456" as previously thought)


def test_decoding_with_repeats():
    """Test the improved decoding function with repeated digits"""
    # Test with repeated digits: 3, 3, 4, 4, blank
    test_tensor = torch.tensor([[
        [0.1, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # 3
        [0.1, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # 3
        [0.1, 0.1, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # 4
        [0.1, 0.1, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # 4
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7],  # blank (index 10)
    ]]).log()  # Convert to log probabilities

    predictions = improved_decode_predictions(test_tensor)
    print(f"Test decoding with repeats: {predictions}")  # Should be "34" (repeats collapsed)


if __name__ == "__main__":
    main()
    test_decoding()
    test_decoding_with_repeats()