import numpy as np

class TestDataGenerator:
    @staticmethod
    def generate_y_data(size, num_classes):
        """
        Generate Y data ensuring all classes are represented.
        
        Args:
            size (int): Size of the output array
            num_classes (int): Number of classes (output_size)
            
        Returns:
            np.array: Array of class labels where all classes are represented
        """
        # First ensure we have at least one sample of each class
        y = np.array(list(range(num_classes)))
        
        # Then fill the rest randomly
        remaining_size = size - num_classes
        if remaining_size > 0:
            random_labels = np.random.randint(0, num_classes, size=remaining_size)
            y = np.concatenate([y, random_labels])
            
        # Shuffle the array to mix the guaranteed classes with random ones
        np.random.shuffle(y)
        return y

    @staticmethod
    def generate_x_data(input_size, num_samples):
        """
        Generate X data for testing.
        
        Args:
            input_size (int): Size of input features
            num_samples (int): Number of samples to generate
            
        Returns:
            np.array: Array of input features
        """
        return np.random.random((input_size, num_samples))

    @staticmethod
    def generate_conv_x_data(num_samples, input_dim, channels=1):
        """
        Generate convolutional X data for testing.
        
        Args:
            num_samples (int): Number of samples
            input_dim (int): Input dimension (assumes square input)
            channels (int): Number of channels
            
        Returns:
            np.array: Array of convolutional input features
        """
        return np.random.random((num_samples, input_dim, input_dim, channels)) 