class UGV:
    m       : float = 1100.0
    Lf      : float = 1.0
    Lr      : float = 1.5
    tw      : float = 3
    Iz      : float | None = None
    Cf      : float = 65000.0
    Cr      : float = 55000.0
    R0      : float = 0.30
    sr      : float = 15.0
    kped    : float = 0.045
    r_ref   : float = 4.0
    kbrk    : float = 0.05 
    cdr     : float = 4.0e-4
    cro     : float = 0.15
    vmin    : float = 1.5
    nsub    : int   = 4

    def __init__(self):
        if self.Iz is None: 
            self.Iz = self.estimate_yaw_inertia()

    def estimate_yaw_inertia(self, dynamic_index=1.0) -> float: 
        return self.m * self.Lf * self.Lr * dynamic_index; 

    def BM_lateral_coefficients(self):
        k1 = (self.Cf + self.Cr) / self.m
        k2 = (self.Lf * self.Cf - self.Lr * self.Cr) / self.m
        j2 = (self.Lf * self.Cf - self.Lr * self.Cr) / self.Iz
        j3 = (self.Lf ** 2 * self.Cf + self.Lr ** 2 * self.Cr) / self.Iz

        return k1, k2, j2, j3
