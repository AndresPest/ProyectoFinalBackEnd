from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


model = load_model('modelo_base.h5')

model.compile(optimizer=Adam(1e-5),  # tasa baja para ajustes finos
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=30,
          callbacks=[tensorboard, reduce_lr])
