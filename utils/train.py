from generator import model, train_data_generator, vald_data_generator, batch_size

epochs = 60

# Start Training
history = model.fit(
    train_data_generator,  steps_per_epoch=train_data_generator.samples // batch_size, epochs=epochs, validation_data=vald_data_generator,
    validation_steps=vald_data_generator.samples // batch_size
)

# Use model.fit_generator() if using TF version &lt; 2.2

