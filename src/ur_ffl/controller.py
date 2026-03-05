class PDController:
    def __init__(self, target_uncertainty=0.4, kp=0.1, kd=0.05, max_severity=1.0):
        self.target_uncertainty = target_uncertainty
        self.kp = kp
        self.kd = kd
        self.max_severity = max_severity
        self.prev_error = 0.0
        self.severity = 0.0

    def compute_severity(self, current_uncertainty):
        error = self.target_uncertainty - current_uncertainty
        p_term = self.kp * error
        d_term = self.kd * (error - self.prev_error)
        
        self.severity += (p_term + d_term)
        self.severity = max(0.0, min(self.severity, self.max_severity))
        self.prev_error = error
        
        return self.severity