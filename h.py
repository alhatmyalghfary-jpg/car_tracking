import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(" (CUDA)")
    for gpu in gpus:
        print(f"GPU: {gpu}")
else:
    print(" (CPU)")