import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from autoaugment import MNISTPolicy  # Import the MNIST-specific autoaugment policy

# Load the CustomMNISTDataset without applying any transforms (for the original image)
train_set = CustomMNISTDataset(root='/content/SRPV2/data', train=True, download=True, apply_transform=False)

# Set up the auto augment transform pipeline
auto_augment_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    MNISTPolicy(),  # AutoAugment policy for MNIST
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize for 3 channels
])

# Choose a single image from the dataset
img, label = train_set[0]  # Selecting the first image for example

# Apply the auto augment transformation
augmented_img = auto_augment_transform(img)

# Convert the tensors to numpy arrays for visualization
def tensor_to_image(tensor):
    tensor = tensor.clone().detach().cpu().numpy()
    tensor = tensor.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
    tensor = tensor * 0.3081 + 0.1307  # Un-normalize using MNIST normalization values
    tensor = tensor.clip(0, 1)  # Clip values to keep them in the range [0, 1]
    return tensor

# Visualize the original and augmented images
original_img_np = tensor_to_image(transforms.ToTensor()(img))
augmented_img_np = tensor_to_image(augmented_img)

# Plotting the images side by side
plt.figure(figsize=(8, 4))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(original_img_np, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Augmented image
plt.subplot(1, 2, 2)
plt.imshow(augmented_img_np, cmap='gray')
plt.title("Augmented Image")
plt.axis('off')

plt.show()
