import torchvision

# Download CIFAR-10 dataset
dataset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True)

# Initialize counters for airplane and dog images
AIRPLANE_COUNT = 0
DOG_COUNT = 0

# Loop through the dataset to find airplane and dog images
for image, label in dataset:
    # Save airplane images
    if label == 0 and AIRPLANE_COUNT < 5:
        image.save(f"airplane_{AIRPLANE_COUNT}.png")
        AIRPLANE_COUNT += 1

    # Save dog images
    if label == 5 and DOG_COUNT < 5:
        image.save(f"dog_{DOG_COUNT}.png")
        DOG_COUNT += 1

    # Break the loop if we have saved 5 images of both categories
    if AIRPLANE_COUNT == 5 and DOG_COUNT == 5:
        break
