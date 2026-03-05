class PDController:
    def __init__(self):
        # Section 3.5.3.1 & 3.5.3.2 parameters
        self.target = 0.15
        self.kp = 0.3
        self.kd = 0.1
        self.min_alpha = 0.3
        self.max_alpha = 0.9
        
        self.prev_error = 0.0
        self.alpha = 0.5 

    def compute_severity(self, mean_zu_sq):
        # Eq 10: e(t) = 0.15 - mean(Zu^2)
        error = self.target - mean_zu_sq
        
        # Eq 11: PD Update Rule
        p_term = self.kp * error
        d_term = self.kd * (error - self.prev_error)
        
        self.alpha += (p_term + d_term)
        
        # Clamping as defined in Section 3.5.3.3
        self.alpha = max(self.min_alpha, min(self.alpha, self.max_alpha))
        self.prev_error = error
        
        return self.alpha