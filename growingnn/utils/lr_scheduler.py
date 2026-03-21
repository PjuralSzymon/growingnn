"""
Learning rate schedulers for neural networks.
"""


class LearningRateScheduler:
    """Learning rate scheduler with different modes."""
    
    CONSTANT = 0
    PROGRESIVE = 1
    PROGRESIVE_PARABOIDAL = 2

    def __init__(self, mode, alpha, steepness=0.2):
        """
        Initialize learning rate scheduler.
        
        Args:
            mode: Scheduler mode (CONSTANT, PROGRESIVE, or PROGRESIVE_PARABOIDAL)
            alpha: Base learning rate
            steepness: Steepness parameter for progressive modes
        """
        self.mode = mode
        self.alpha = alpha
        self.steepness = steepness

    def alpha_scheduler(self, i, iterations):
        """
        Calculate learning rate for given iteration.
        
        Args:
            i: Current iteration
            iterations: Total iterations
            
        Returns:
            Learning rate for current iteration
        """
        thresh = float(self.steepness * iterations)
        i = float(i)
        iterations = float(iterations)
        result = self.alpha
        
        if self.mode == LearningRateScheduler.CONSTANT:
            result = self.alpha
        elif self.mode == LearningRateScheduler.PROGRESIVE:
            if i < thresh:
                return self.alpha * ((i+1) / (thresh + 2))
            result = self.alpha * (1 - (i - thresh) / (iterations - thresh + 2))
        else:
            thresh = self.steepness * iterations
            if i < thresh:
                return self.alpha * ( -1 * ((1) / (pow(thresh, 2))) * pow((i) - thresh, 2) + 1)
            result = self.alpha * (-1 * ((1) / (pow(iterations - thresh, 2))) * pow((i) - thresh, 2) + 1)
        
        return max(0, result)
