class DegradationSelector:
    def select(self, z_u_scores):
        selections = []
        for zu in z_u_scores:
            zu_val = zu.item()
            # Table 9 Logic
            if zu_val < -1.0:
                selections.append('smear')
            elif -1.0 <= zu_val < 0.0:
                selections.append('ripple')
            elif 0.0 <= zu_val <= 1.0:
                selections.append('quantize')
            else:
                selections.append('noise')
        return selections