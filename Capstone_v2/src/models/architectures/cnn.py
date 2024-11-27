import tensorflow as tf
from tensorflow.keras import layers, Model, Input

def create_cnn(config, **kwargs):
    """Create CNN model """
    # Get input shape from config
    input_shape = config.IMG_SHAPE
    
    # Get model parameters from kwargs
    model_params = kwargs.get('model_params', {})
    num_conv_layers = model_params.get('num_conv_layers', 3)
    num_filters = model_params.get('num_filters', 32)
    
    # Convert kernel_size from string to tuple
    kernel_size_str = model_params.get('kernel_size', "3x3")
    if isinstance(kernel_size_str, str):
        k = int(kernel_size_str.split('x')[0])
        kernel_size = (k, k)
    else:
        kernel_size = model_params.get('kernel_size', (3, 3))
    
    pool_type = model_params.get('pool_type', 'max')
    dropout_rate = model_params.get('dropout_rate', 0.5)
    learning_rate = model_params.get('learning_rate', 0.001)
    
    # Create model
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Add convolutional layers
    for i in range(num_conv_layers):
        x = layers.Conv2D(
            num_filters * (2 ** i),  # Double filters each layer
            kernel_size,
            padding='same',
            activation='relu'
        )(x)
        x = layers.BatchNormalization()(x)
        
        # Add pooling layer with proper pool_size
        if pool_type == 'max':
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        else:
            x = layers.AveragePooling2D(pool_size=(2, 2))(x)
            
        x = layers.Dropout(dropout_rate)(x)
    
    # Add classification layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='CNN')
    
    # Compile model
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
