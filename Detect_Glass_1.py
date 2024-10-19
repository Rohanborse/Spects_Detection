import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

# 1. Load MobileNetV2 pre-trained model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

# 2. Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# 3. Build the model
model = Model(inputs=base_model.input, outputs=predictions)

# Load previously saved model if available
import os
checkpoint_path = 'glasses_detector_checkpoint.keras'
if os.path.exists(checkpoint_path):
    print(f"Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path)
else:
    print("No saved model found, starting from scratch")

# 4. Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Prepare the dataset
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'F:/Spects_wear_or_not/dataset/',  # Update with your dataset path
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    'F:/Spects_wear_or_not/dataset/',  #
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation')

# 6. Callbacks for saving progress
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=False, save_weights_only=False, mode='auto', verbose=1)
csv_logger = CSVLogger('training_log.csv', append=True)

# 7. Train the model with frozen layers
model.fit(train_generator, validation_data=validation_generator, epochs=5,
          steps_per_epoch=train_generator.samples // 32, validation_steps=validation_generator.samples // 32,
          callbacks=[checkpoint, csv_logger])

# 8. Fine-tuning: Unfreeze the base model for further training
base_model.trainable = True  # Unfreeze the entire base model

# Recompile the model with a lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Continue training with fine-tuning
model.fit(train_generator, validation_data=validation_generator, epochs=5,
          steps_per_epoch=train_generator.samples // 32, validation_steps=validation_generator.samples // 32,
          callbacks=[checkpoint, csv_logger])

# 9. Final model save
model.save('glasses_detector_final_finetuned.keras')
print("Model fine-tuning complete and saved as 'glasses_detector_final_finetuned.keras'")
