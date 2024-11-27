# src/data/data_loader.py

import os
import numpy as np
import tensorflow as tf

class DataLoader:
    """Class to handle data loading and preprocessing"""
    def __init__(self, config):
        self.config = config
        self.image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )

    def load_datasets(self):
        """Load and combine all datasets"""
        combined_data = {
            'train': {'images': [], 'labels': []},
            'val': {'images': [], 'labels': []},
            'test': {'images': [], 'labels': []}
        }

        for dataset in self.config.DATASETS:
            dataset_path = os.path.join(self.config.BASE_PATH, dataset)
            for split in ['train', 'validation', 'test']:
                print(f"Loading {split} data from {dataset}...")
                
                ds = self.image_generator.flow_from_directory(
                    os.path.join(dataset_path, split if split != 'val' else 'validation'),
                    target_size=self.config.IMG_SIZE,
                    batch_size=self.config.BATCH_SIZE,
                    class_mode='binary',
                    shuffle=True
                )
                
                # Calculate total number of samples
                total_samples = ds.n
                
                # Apply dataset fraction if enabled
                if self.config.USE_DATASET_FRACTION:
                    num_samples = int(total_samples * self.config.DATASET_FRACTION)
                    num_batches = int(num_samples / self.config.BATCH_SIZE)
                    print(f"Using {self.config.DATASET_FRACTION * 100}% of {split} data: {num_samples} samples")
                else:
                    num_batches = int(total_samples / self.config.BATCH_SIZE)
                    print(f"Using full {split} dataset: {total_samples} samples")
                
                # Load batches
                images, labels = [], []
                for _ in range(num_batches):
                    batch_images, batch_labels = next(ds)
                    images.append(batch_images)
                    labels.append(batch_labels)
                
                split_key = 'val' if split == 'validation' else split
                if images and labels:
                    combined_data[split_key]['images'].append(np.concatenate(images))
                    combined_data[split_key]['labels'].append(np.concatenate(labels))
                else:
                    print(f"Warning: No data loaded for {split} split")

        # Combine all datasets
        train_data = (np.concatenate(combined_data['train']['images']), 
                     np.concatenate(combined_data['train']['labels']))
        val_data = (np.concatenate(combined_data['val']['images']), 
                   np.concatenate(combined_data['val']['labels']))
        test_data = (np.concatenate(combined_data['test']['images']), 
                    np.concatenate(combined_data['test']['labels']))

        # Convert to float32
        train_data = (train_data[0].astype('float32'), train_data[1].astype('float32'))
        val_data = (val_data[0].astype('float32'), val_data[1].astype('float32'))
        test_data = (test_data[0].astype('float32'), test_data[1].astype('float32'))

        print("\nFinal Dataset Statistics:")
        print(f"Training set: {train_data[0].shape[0]} samples")
        print(f"    Shape: {train_data[0].shape}")
        print(f"    Class distribution: {np.bincount(train_data[1].astype(int))}")
        print(f"Validation set: {val_data[0].shape[0]} samples")
        print(f"    Shape: {val_data[0].shape}")
        print(f"    Class distribution: {np.bincount(val_data[1].astype(int))}")
        print(f"Test set: {test_data[0].shape[0]} samples")
        print(f"    Shape: {test_data[0].shape}")
        print(f"    Class distribution: {np.bincount(test_data[1].astype(int))}")

        return train_data, val_data, test_data
