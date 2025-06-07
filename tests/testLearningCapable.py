import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import unittest
from testSuite import mode
from unittest.mock import MagicMock
import numpy as np
import matplotlib.pyplot as plt
from growingnn.structure import History
import json

# Best settings found from previous runs
PATIENCE = 10
VERBOSE = 0.5

class TestLearningCapable(unittest.TestCase):
    def setUp(self):
        self.history = History(['accuracy', 'loss'])
        self.visualize = True  # Set to True to generate plots
        self.patience = PATIENCE
        self.verbose = VERBOSE
        
    def test_spectrum_of_learning_curves(self):
        """Test that visualizes a spectrum of learning curves from fast-then-slow to slow-then-fast"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create figure
        plt.figure(figsize=(20, 15))
        
        # Generate many different learning curves
        num_curves = 100
        epochs = 50
        
        # Create different types of curves
        for i in range(num_curves):
            # Randomly choose curve type
            curve_type = np.random.choice(['fast_slow', 'slow_fast', 'oscillating', 'step', 'sigmoid', 'plateau', 'fluctuating'])
            
            if curve_type == 'fast_slow':
                # Fast then slow learning
                fast_ratio = np.random.uniform(0.3, 0.7)
                fast_epochs = int(epochs * fast_ratio)
                slow_epochs = epochs - fast_epochs
                
                fast_improvement = np.linspace(0.5, 0.8, fast_epochs)
                slow_improvement = np.linspace(0.8, 0.81, slow_epochs)  # Very small improvement
                curve = np.concatenate([fast_improvement, slow_improvement])
                
            elif curve_type == 'slow_fast':
                # Slow then fast learning
                slow_ratio = np.random.uniform(0.3, 0.7)
                slow_epochs = int(epochs * slow_ratio)
                fast_epochs = epochs - slow_epochs
                
                slow_improvement = np.linspace(0.5, 0.51, slow_epochs)  # Very small improvement
                fast_improvement = np.linspace(0.51, 0.85, fast_epochs)
                curve = np.concatenate([slow_improvement, fast_improvement])
                
            elif curve_type == 'oscillating':
                # Oscillating pattern with overall improvement
                base = np.linspace(0.5, 0.55, epochs)  # Small overall improvement
                oscillations = 0.05 * np.sin(np.linspace(0, 8*np.pi, epochs))
                curve = base + oscillations
                
            elif curve_type == 'step':
                # Step-like improvements
                steps = np.random.randint(3, 8)
                step_sizes = np.random.uniform(0.001, 0.005, steps)  # Smaller steps
                curve = np.zeros(epochs)
                current = 0.5
                for step in range(steps):
                    step_length = epochs // steps
                    start = step * step_length
                    end = start + step_length if step < steps-1 else epochs
                    current += step_sizes[step]
                    curve[start:end] = current
                    
            elif curve_type == 'sigmoid':
                # Sigmoid-like curve
                x = np.linspace(-6, 6, epochs)
                curve = 0.5 + 0.35 * (1 / (1 + np.exp(-x)))
                
            elif curve_type == 'plateau':
                # Plateau pattern
                plateau_value = np.random.uniform(0.5, 0.7)
                curve = np.ones(epochs) * plateau_value
                
            else:  # fluctuating
                # Fluctuating pattern
                base = np.ones(epochs) * 0.5
                fluctuations = 0.05 * np.sin(np.linspace(0, 20*np.pi, epochs))
                curve = base + fluctuations
            
            # Add some noise to make it more realistic
            noise = np.random.normal(0, 0.003, len(curve))
            curve = curve + noise
            
            # Ensure curve stays within reasonable bounds
            curve = np.clip(curve, 0.45, 0.95)
            
            # Check if pattern is learning capable with current parameters
            is_learning = History._check_learning_capability(curve, patience=self.patience, verbose=self.verbose)
            
            # Plot with appropriate color
            color = 'green' if is_learning else 'red'
            plt.plot(curve, color=color, linewidth=1, alpha=0.3, 
                    label=f'Curve {i+1} ({curve_type})' if i < 10 else None)
        
        plt.title(f'Spectrum of Learning Curves\nGreen = Still Learning, Red = Not Learning\nPatience={self.patience}, Verbose={self.verbose}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        
        # Add threshold lines
        threshold = self.verbose  # Based on verbose
        plt.axhline(y=0.5 + threshold, color='gray', linestyle='--', alpha=0.5)
        plt.axhline(y=0.5 - threshold, color='gray', linestyle='--', alpha=0.5)
        
        # Add legend (only for first 10 curves to avoid clutter)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'learning_curves_spectrum_p{self.patience}_v{self.verbose}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Test should pass as it's just visualization
        self.assertTrue(True)

if __name__ == '__main__':
    # Run tests with best settings
    print(f"\nRunning tests with best settings: patience={PATIENCE}, verbose={VERBOSE}")
    unittest.main(verbosity=2)