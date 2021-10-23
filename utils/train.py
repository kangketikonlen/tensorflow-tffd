import matplotlib.pyplot as plt
from generator import model, train_data_generator, vald_data_generator, batch_size

epochs = 5

# Start Training
history = model.fit(
    train_data_generator,  steps_per_epoch=train_data_generator.samples // batch_size, epochs=epochs, validation_data=vald_data_generator,
    validation_steps=vald_data_generator.samples // batch_size
)

# Use model.fit_generator() if using TF version &lt; 2.2

# Plot the accuracy and loss curves for both training and validation

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training loss')
plt.legend()

plt.show()
