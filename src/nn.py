import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, input_size=(16, 16), num_classes=10):
        """
        Args:
            input_channels (int): Number of channels in the input image (default is 1 for grayscale).
            input_size (tuple): Spatial dimensions of the input image as (height, width).
            num_classes (int): Number of classes for the classification output.
        """
        super(SimpleCNN, self).__init__()
        
        # First convolutional layer: from input_channels to 16 channels.
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
        # Second convolutional layer: from 16 channels to 32 channels.
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # Max pooling layer: halves the spatial dimensions.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # For a 16x16 input:
        # After the first pooling: 16x16 -> 8x8
        # After the second pooling: 8x8 -> 4x4
        final_height = input_size[0] // 4  # 16 // 4 = 4
        final_width  = input_size[1] // 4  # 16 // 4 = 4
        
        # Fully connected layers:
        # The input features to fc1: (number of channels from conv2) * (final_height) * (final_width)
        self.fc1 = nn.Linear(32 * final_height * final_width, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x, penultimate=False, flattened=False):
        if flattened:
            imagesize = int(x.shape[1] ** 0.5)
            x = x.view(-1, 1, imagesize, imagesize)
            
        # Apply first convolution, ReLU activation, and pooling.
        x = self.pool(F.relu(self.conv1(x)))
        # Apply second convolution, ReLU activation, and pooling.
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the tensor from (batch_size, channels, height, width) to (batch_size, -1)
        x = x.view(x.size(0), -1)
        # First fully connected layer with ReLU activation.
        x = F.relu(self.fc1(x))
        
        if penultimate:
            return x
        else:
            # Final output layer producing logits for each class.
            x = self.fc2(x)
            return x

# Example usage:
if __name__ == '__main__':
    # Create the model for a single-channel, 16x16 image, and 10 classes.
    model = SimpleCNN(input_channels=1, input_size=(16, 16), num_classes=1)
    print(model)
    
    # Create a dummy input tensor with batch size = 4.
    dummy_input = torch.randn(4, 1, 16, 16)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected output shape: torch.Size([4, 10])
