import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image

# Path to your dataset
dataset_path = "C:/Users/slive/Downloads/archive (8)/leaves"

# Define image size and batch size
image_size = (224, 224)  # MobileNetV2 requires images of size (224, 224)
batch_size = 32

# Create ImageDataGenerators for loading the images from the folders
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # rescale pixel values to [0, 1]

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # Multi-class classification
    subset='training'  # Use 80% for training
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Use 20% for validation
)

# Load the MobileNetV2 model with pre-trained weights (excluding the top classification layers)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze the layers of MobileNetV2 (so that they aren't trained again)
base_model.trainable = False

# Add custom layers on top for classification
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  # 4 classes: curl, spot, healthy, slug
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Save the model if needed
model.save('leaf_disease_classifier_mobilenet.h5')

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')
