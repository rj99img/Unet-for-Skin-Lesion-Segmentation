# Unet-for-Skin-Lesion-Segmentation
The following contains the segmentation Code of U-Net ResNet 50 for Skin Lession Segmentation 
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import classification_report, jaccard_score
import matplotlib.pyplot as plt
import random

# Set paths to the directories
train_dir = 'TRAIN'
truth_dir = 'TRUTH'
val_train_dir = 'VAL_TRAIN'
val_truth_dir = 'VAL_TRUTH'

# Parameters
img_size = (128, 128)  # Adjust as needed

# Load and preprocess images
def load_images(train_dir, truth_dir, img_size):
    train_images = []
    truth_images = []

    train_files = os.listdir(train_dir)

    for filename in train_files:
        train_path = os.path.join(train_dir, filename)

        # Modify the filename to match the ground truth file
        truth_filename = filename.replace('.jpg', '_segmentation.png')
        truth_path = os.path.join(truth_dir, truth_filename)

        if not os.path.isfile(truth_path):
            print(f"Warning: Ground truth image {truth_path} not found")
            continue

        train_image = cv2.imread(train_path, cv2.IMREAD_GRAYSCALE)
        truth_image = cv2.imread(truth_path, cv2.IMREAD_GRAYSCALE)

        if train_image is None:
            print(f"Warning: Could not read train image {train_path}")
            continue

        if truth_image is None:
            print(f"Warning: Could not read truth image {truth_path}")
            continue

        train_image = cv2.resize(train_image, img_size)
        truth_image = cv2.resize(truth_image, img_size)

        # Convert grayscale images to RGB by duplicating the channel
        train_image = cv2.cvtColor(train_image, cv2.COLOR_GRAY2RGB)

        train_images.append(train_image)
        truth_images.append(truth_image)

    train_images = np.array(train_images, dtype=np.float32) / 255.0
    truth_images = np.array(truth_images, dtype=np.float32) / 255.0

    truth_images = np.expand_dims(truth_images, axis=-1)

    return train_images, truth_images

# Load the images
train_images, truth_images = load_images(train_dir, truth_dir, img_size)
val_train_images, val_truth_images = load_images(val_train_dir, val_truth_dir, img_size)

def unet_resnet50(input_shape):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    # Freeze the base model
    base_model.trainable = False

    inputs = tf.keras.Input(input_shape)

    # Encoder: ResNet50
    s1 = base_model.get_layer("conv1_relu").output
    s2 = base_model.get_layer("conv2_block3_out").output
    s3 = base_model.get_layer("conv3_block4_out").output
    s4 = base_model.get_layer("conv4_block6_out").output

    # Bottleneck
    b1 = base_model.get_layer("conv5_block3_out").output

    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(b1)
    u6 = layers.concatenate([u6, s4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, s3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, s2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, s1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    u10 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c9)
    c10 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u10)
    c10 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c10)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c10)

    model = Model(inputs=[base_model.input], outputs=[outputs])
    return model

# Define the model
input_shape = (img_size[0], img_size[1], 3)  # Adjusted to 3 channels (RGB)
model = unet_resnet50(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Training the model
history = model.fit(
    train_images, truth_images,
    epochs=300,
    batch_size=32,
    validation_data=(val_train_images, val_truth_images)
)

# Save the model
model.save('unet_resnet50_model.h5')

# Evaluate the model
val_predictions = model.predict(val_train_images)
val_predictions = (val_predictions > 0.5).astype(np.uint8)

# Flatten the predictions and labels for classification report
val_predictions_flat = val_predictions.flatten().astype(int)
val_truth_images_flat = val_truth_images.flatten().astype(int)

# Classification report
print("Classification Report:")
print(classification_report(val_truth_images_flat, val_predictions_flat, target_names=['Background', 'Object']))

# Jaccard index
jaccard = jaccard_score(val_truth_images_flat, val_predictions_flat)
print(f"Jaccard Index: {jaccard}")

# Plot accuracy and loss graphs
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

plot_history(history)

# Display a random validation image along with its ground truth and predicted masks
def display_random_validation_image(val_train_images, val_truth_images, val_predictions):
    idx = random.randint(0, len(val_train_images) - 1)
    original_image = val_train_images[idx]
    ground_truth = val_truth_images[idx]
    predicted_mask = val_predictions[idx]

    jaccard = jaccard_score(ground_truth.flatten(), predicted_mask.flatten())
    print("SCORE")
    print(jaccard)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(ground_truth.squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Predicted Mask\nJaccard Index: {jaccard:.4f}")
    plt.imshow(predicted_mask.squeeze(), cmap='gray')
    plt.axis('off')

    plt.show()

display_random_validation_image(val_train_images, val_truth_images, val_predictions)

# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the trained UNet ResNet50 model
model = tf.keras.models.load_model('unet_resnet50_model.h5')

# Parameters
img_size = (128, 128)

def predict_segmentation(image, original_size):
    # Preprocess the image
    image_resized = cv2.resize(image, img_size)
    image_resized = image_resized.astype(np.float32) / 255.0
    image_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
    image_input = np.expand_dims(image_resized, axis=0)

    # Predict the segmentation mask
    prediction = model.predict(image_input)
    segmented_image = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255

    # Resize the segmented image back to original size
    segmented_image_resized = cv2.resize(segmented_image, original_size)

    return segmented_image_resized

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_size = (image.shape[1], image.shape[0])  # (width, height)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Predict segmentation
    segmented_image = predict_segmentation(image, original_size)

    # Find contours
    contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes on the original image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Plotting
    plt.figure(figsize=(12, 6))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Segmented image
    plt.subplot(1, 3, 2)
    plt.imshow(segmented_image, cmap='gray')
    plt.title('Segmented Image')
    plt.axis('off')

    # Bounding box image
    plt.subplot(1, 3, 3)
    plt.imshow(image_rgb)
    plt.title('Bounding Box Image')
    plt.axis('off')

    # Show plot
    plt.tight_layout()
    plt.show()

# Example usage
image_path = '/content/TRAIN/ISIC_0012623.jpg'
process_image(image_path)

