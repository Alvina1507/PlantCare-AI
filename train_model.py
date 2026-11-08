
import tensorflow as tf
import json
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 123

data_dir = r"C:\Users\alvin\PlantCareAI\PlantVillageData\Plant_leave_diseases_dataset_without_augmentation"

# ==========================================
# 2. LOAD TRAINING DATASET
# ==========================================

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# ==========================================
# 3. LOAD VALIDATION DATASET
# ==========================================

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# ==========================================
# 4. GET CLASS NAMES
# ==========================================

class_names = train_dataset.class_names
num_classes = len(class_names)

print("\nNumber of classes:", num_classes)

# Save class names
with open("class_names.json", "w") as file:
    json.dump(class_names, file)

# ==========================================
# 5. OPTIMIZE DATASET
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# ==========================================
# 6. LOAD MOBILENETV2
# ==========================================

print("\nLoading MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# ==========================================
# 7. BUILD MODEL
# ==========================================

inputs = tf.keras.Input(shape=(224, 224, 3))

# Preprocess images
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

# Extract features
x = base_model(x, training=False)

# Reduce dimensions
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Dropout to reduce overfitting
x = tf.keras.layers.Dropout(0.3)(x)

# Classification layer
outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

# ==========================================
# 8. COMPILE MODEL
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel created successfully!")

# ==========================================
# 9. DISPLAY MODEL SUMMARY
# ==========================================

model.summary()

# ==========================================
# 10. CALLBACKS
# ==========================================

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_plant_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]

# ==========================================
# 11. TRAIN MODEL
# ==========================================

print("\nStarting model training...")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==========================================
# 12. EVALUATE MODEL
# ==========================================

print("\nEvaluating model...")

loss, accuracy = model.evaluate(validation_dataset)

print("\nValidation Loss:", loss)
print("Validation Accuracy:", accuracy * 100, "%")

# ==========================================
# 13. SAVE TRAINED MODEL
# ==========================================

model.save("plant_disease_model.keras")

print("\n================================")
print("PLANTCARE AI TRAINING COMPLETED")
print("================================")

print("Model saved successfully!")
print("Classes:", num_classes)
print("Model file: plant_disease_model.keras")
print("Class file: class_names.json")
