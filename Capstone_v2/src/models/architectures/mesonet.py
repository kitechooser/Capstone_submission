import tensorflow as tf
from tensorflow.keras import layers, Model, Input

def create_mesonet(config, **kwargs):
    """Create MesoNet model"""
    # Get input shape from config
    input_shape = config.IMG_SHAPE
    
    # Get model parameters from kwargs
    model_params = kwargs.get('model_params', {})
    
    # Extract architecture parameters with defaults
    initial_filters = model_params.get('initial_filters', 16)
    conv1_size = model_params.get('conv1_size', 3)
    conv_other_size = model_params.get('conv_other_size', 5)
    dropout_rate = model_params.get('dropout_rate', 0.5)
    dense_units = model_params.get('dense_units', 32)
    leaky_relu_alpha = model_params.get('leaky_relu_alpha', 0.1)
    learning_rate = model_params.get('learning_rate', 0.0001)
    
    # Extract training parameters
    batch_size = model_params.get('batch_size', config.BATCH_SIZE)
    
    # Create input layer
    inputs = Input(shape=input_shape)
    
    # Conv Block 1 - using conv1_size
    x = layers.Conv2D(initial_filters, (conv1_size, conv1_size), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Conv Block 2 - using conv_other_size
    x = layers.Conv2D(initial_filters * 2, (conv_other_size, conv_other_size), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Conv Block 3 - using conv_other_size
    x = layers.Conv2D(initial_filters * 4, (conv_other_size, conv_other_size), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Conv Block 4 - using conv_other_size
    x = layers.Conv2D(initial_filters * 4, (conv_other_size, conv_other_size), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((4, 4), padding='same')(x)
    
    # Dense Layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units)(x)
    x = layers.LeakyReLU(negative_slope=leaky_relu_alpha)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='MesoNet')
    
    # Compile model with proper learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )
    
    return model
