
import tensorflow as tf

data_dir = r"C:\Users\alvin\PlantCareAI\PlantVillageData\Plant_leave_diseases_dataset_without_augmentation"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names

print("\nDataset loaded successfully!")
print("Number of classes:", len(class_names))
print("Class names:", class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

print("\nTraining and validation datasets are ready!")
