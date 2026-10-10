#for part 4 depends on python version 
#from __future__ import annotation 

#for tewzi3 l data bel functions
# @dataclass shortcut for glodal and coupling paramaters 
from dataclasses import dataclass

# for arrays and matrices 
import numpy as np 

#block_diag place matrices along diagonal
#from scipy.linalg import block_diag 
from scipy.linalg import block_diag

#vehicale model
import ssm1 as ssm

VehicleSSM = ssm.VehicleSSM

N_STATE = len(ssm.STATE_NAMES) #11 vehicle states
N_OBS =len(ssm.OBS_NAMES)      #13 measurements /sensors

# Positions of the states we use inside the 11-number vector
#we will choose 6 only cz no need for others they are handled by ssm 
IDX_PX = ssm.STATE_NAMES.index("px") # =0 east position [m]
IDX_PY = ssm.STATE_NAMES.index("py") # =1 north position [m] 
IDX_PSI = ssm.STATE_NAMES.index("psi") # =2 heading angle [rad]
IDX_VX =ssm.STATE_NAMES.index("vx") # =3 longitudinal or forward velocity [m/s]
IDX_VY =ssm.STATE_NAMES.index("vy") # =4 lateral or sideways velocity [m/s]
IDX_DELTA =ssm.STATE_NAMES.index("delta") # =10 steering angle [rad]


#GLOBAL TRAFFIC STATE G 
#Layer A

GLOBAL_NAMES = ["density_veh_km", "flow_veh_h", "mean_speed_mps", "congestion"] #4 global traffic states

#Number of global states
N_GLOBAL = len(GLOBAL_NAMES) #4 global traffic states

#Position inside G of the 2 values the vehicles react to.
IDX_MEAN_SPEED = GLOBAL_NAMES.index("mean_speed_mps") # =2 mean speed [m/s]
IDX_CONGESTION = GLOBAL_NAMES.index("congestion") # =3 congestion [0-1]



@dataclass
class GlobalParams:
    """Layer A: how the global traffic states G changes over time."""

    #Small tau = G reacts fast to the car. Large tau = G reacts slowly to the car.
    #tau(TheGreekletterTau): time constant for the global traffic states G [s]
    tau: float = 2.0 #[seconds]

    #Speed of completely free traffic. Used to compute congestion.
    v_free: float = 25.0 #[m/s]

    #Distance between vehicles only 1 V2V
    fallback_spacing : float = 500.0 #[m]

    #Random noise added to the global traffic states G.
    noise_std: tuple = (2.0, 50.0, 0.2, 0.02)


@dataclass
class CouplingParams:
    #How the global state and the neighbours push on each car.

    #macro: a car is pulled toward the average speed of traffic.
    gain_macro: float = 0.15 # pull toward mean speed[1/s]

    #The weight W_ij: how much does car j influence car i.
    range_m: float = 40.0 #[m]how far ahead a car still matters   
    lane_sigma: float = 2.0 #[m] sideways tolerance for "same line"
    ahead_softness: float = 2.0 #[m] ssmoothness of the "is ahead switch"

    #the reaction phi_ij: a simple car-following role
    gain_speed: float = 0.5 #[1/s] react to the speed difference
    gain_gap: float = 0.05 #[1/s^2] react to the gap error
    standstill_gap: float = 5.0 #[m] the gap a driver wants at zero speed
    time_headway: float = 1.2 #[s] extra gap wanted per m/s of speed

    # The total push
    a_max: float = 3.0 #[m/s^2] maximum acceleration


#PART 3: GRAPH STRUCTURE (THE ARROW OF DBN)
#THE NODES NAMES ARE:
#G[t] = global traffic states at time t
#x1[t] = state of vehicle 1 at time t
#y1[t] = observation(sensors) of vehicle 1 at time t
#u1[t] = driver input of vehicle 1 at time t

# an arrow is written as pair: (parent, child) = (from, to)


def two_slice_edges(n_vehicles):
    #Return the list of all arrows betweentime t-1 and time t.
    edges = [("G[t-1]", "G[t]")]  # Global state G[t-1] influences G[t]
    # Global state G[t] influences all vehicle states x_i[t]

    for i in range(1,n_vehicles +1):
        #f"x{i}[t-1]" is an f-string with i = 2 it becomes the text "x2[t-1]"
        edges+=[
            (f"x{i}[t-1]",f"G[t]"),  # car -> global (G is built from the cars)
            ("G[t]", f"x{i}[t]"),  # global -> car (the car reacts to the traffic)
            (f"x{i}[t-1]", f"x{i}[t]"),  # The car remembers its own past
            (f"u{i}[t-1]", f"x{i}[t]"),  # driver inputs move the car
            (f"x{i}[t]", f"y{i}[t]"),  # The sensors measure the car's state
            (f"u{i}[t]", f"y{i}[t]"),  # inputs also influence or reach the sensors (e.g. brake lights)
        ]

        #Arrows from every OTHER car j into car i (the interaction W_ij)
        #builds a list. "if j != i" skips the car itself.
        edges += [
            (f"x{j}[t-1]", f"x{i}[t]") 
            for j in range(1, n_vehicles + 1) 
            if j != i
        ]
    return edges


def is_acyclic(edges):
    #Check if the graph is acyclic (no loops).
    #Keep node have no parents left.
    
    #The set of all nodes in the graph names
    nodes = {name for edge in edges for name in edge}

    # in_degree[node]: how many parents does each node have 
    #dict.fromkeys(nodes,0) makes dictionary: every node starts at 0.
    in_degree = dict.fromkeys(nodes, 0)

    # Calculate in-degrees
    for _, child in edges:  # "_" means "I don't care about the parent name or need the value"
        in_degree[child] += 1 #count one more parent for this child 

    #nodes with no parents left (in_degree = 0) can be removed first
    ready = [name for name in nodes if in_degree[name] == 0]

    removed = 0
    while ready:      # repeat while the list is note empty
        node = ready.pop()  # take one node out of the list
        removed += 1
        # Removing this node frees its childrem from one parent of each
        for parent, child in edges:
            if parent == node:
                in_degree[child] -= 1
                if in_degree[child] == 0: #no parents left:it can go too
                    ready.append(child) 

    #if every node was removed, there is no loop.
    return removed == len(nodes)


#PART 4: THE TRAFFIC DYNAMIC BAYESIAN NETWORK (DBN) CLASS (skeleton)


#The model keep ONE  long vector for the whole system
# ex: z= [ G (4 numbers) | car 1 (11) | car 2 (11) | car 3 (11) ]
#so with 3 cars: 4 + 3*11 = 37 numbers in the vector z

class TrafficDBN:
    #Joint model of the global traffic states G and N VEHICLES 

    def __init__ (self,vehicles,glob=None, coupling=None):
        #vehicles: list of VehicleSSM objects, one for each car
        self.vehicles = list(vehicles) # the list of VehicleSSM objects
        self.N = len(self.vehicles) #how many cars
        self.dt = self.vehicles[0].dt  # time step in seconds (from the SSM)

        #global traffic states G
        self.glob = glob or GlobalParams()

        #how the global state and the neighbours push on each car
        self.cpl = coupling or CouplingParams()

        # size of the joint vector z: G plus 11 numbers per car 
        self.dim = N_GLOBAL + self.N * N_STATE


        #NOTES:
        #noise matrices (used by thr simulation and the filter)
        # Process noise of G: a diagonal matrix with with the variance (std squared)
        #of each G variable, scaled by the time step.
        q_global = np.diag(np.square(self.glob.noise_std)) * self.dt

        # Joint process noise Q: blocks along the daigonal, one for G, and one per car.
        #self.course_rows = ssm.IDX_COURSE + N_OBS * np.arange(self.N)

# i have problem in the line before so i have fix it from ai with the 3 lines bellow 
                # Joint process noise Q: blocks along the diagonal, one for G, and one per car.
        self.Q = block_diag(q_global, *[v.Q * self.dt for v in self.vehicles])

        # Joint measurement noise R: one block per car (G has no sensors).
        self.R = block_diag(*[v.R for v in self.vehicles])

        # Which rows of the measurement vector are ANGLES (the GPS course).
        self.course_rows = ssm.IDX_COURSE + N_OBS * np.arange(self.N)



        #joint measurement noise R: one block per car (G has no sensor)
        self.R = block_diag(*[v.R for v in self.vehicles])



    #Moving between "G and a table of cars" and "one long vector z"

    def pack(self,G,X):
        #note: G(4,) and X (n,11) -> one vector z of length 4 + 11 N.
        #np.reval flattens the table X row by row; np.concatenate joins.
        return np.concatenate([G, np.ravel(X)])   

    def unpack(self, z):
        # 1 vector z -> G(4,) and X(V,11).
        G = z[:N_GLOBAL]  # the first 4 numbers 
        X = z[N_GLOBAL:].reshape(self.N,N_STATE) # the reset, as a table
        return G, X

    def state_slice(self, i):
        # wheere car i (0,1,2,...) sites inside z,as a slice.
        start = N_GLOBAL + i * N_STATE
        return slice(start, start + N_STATE)

    """
    example:
       car 0 → positions 4 to 14
       car 1 → positions 15 to 25
       car 2 → positions 26 to 36
    """


# Part 5: Layerr A (Global Trafic State G)

    def macro_summary(self, X):
        #Look at all cars and compute the 4 trafic numbers from them 
        # X:  table of car state,shape(N,11)
        #returns: array [density, flow, mean_speed, congestion]

        g = self.glob

        #mean speed, Ground speed of each car from its 2 speed component
        # X[;, IDX_VX] means "column IDX_V of every row" = all forward speed

        speeds = np.hypot(X[:, IDX_VX], X[:, IDX_VY])
        mean_speed = speeds.mean()  #average over all cars

        #Density: how close together are the cars?
        if self.N > 1:
            #positions of all cars, shape (N, 2): east and north
            pos = X[:, [IDX_PX, IDX_PY]]
            # Distance between every pair of cars,as an NxN table
            #pos[:, None, :] has shape (N,1,2) and pos[None, :, :] has shape
            #(1,N,2). Subtracting them pairs every car with every other car then turning into a distance
            dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)

            # A car is at distance 0 from itself. Set the diagonal to infinity 
            # so a car is never counted as its own nearest neighbour.

            np.fill_diagonal(dist, np.inf)

            # For each car, the distance to its nearest neighbour (min over each
            # row), then the average over all cars.
            spacing = dist.min(axis=1).mean()

        else:     
            # with one car there is no neighbour to measure 
            spacing = g.fallback_spacing 

        # max (spacing, 1.0) avoids dividing by almost zero if two cars overlap.
        density = 1000.0 / max(spacing, 1.0)    # cars per kilometer
        # flow: the classic relation flow = density x speed
        # the 3.6 converts m/s to km/h, so the flow comes out in car per hour
        flow = density * mean_speed * 3.6

        # congestion: 0 = free road, 1 = stopped traffic
        # np.clip keep the value between 0 and 1 
        congestion = np.clip(1.0 - mean_speed / g.v_free, 0.0, 1.0)
        return np.array([density, flow, mean_speed, congestion])



    def global_step(self, G, X):

        #one time step of G (without noise).
        # G moves a small part of the way toward what the cars show write now.
        # The part is dt / tau.

        target = self.macro_summary(X)
        return G + (self.dt / self.glob.tau) * (target - G)



    #Part 6: Interaction between cars V2V

    def influence_weights(self, X):
        """How much does car j influence car i?
        
                returns W   : table (N, N), W[i, j] between 0 and 1
                              high if car j is AHEAD of i, in the SAME LANE, and CLOSE
                        gap : table (N, N), gap[i, j] = distance from i to j measured
                              along the direction car i is facing (+ means j is ahead)
                """

        c = self.cpl   #heading of each car
        psi = X[:, IDX_PSI]

        # offset[i, j] = pposition of car j minus position of car i 
        pos = X[:,(IDX_PX, IDX_PY)]
        offset = pos[None, :,:] - pos[:, None, :] # shape (N,N,2)

        #Two unit arrows for each car i: where it faces, and its left side.
        forward = np.stack([np.cos(psi),np.sin(psi)],axis=1) #(N, 2)
        left = np.stack([-np.sin(psi),np.cos(psi)],axis=1) #(N, 2)

        #Split every offset into "along my direction" abd "sideways"
        #(offset * forward[:,None,:]) multiply the x and. y parts and
        #.sum(axis=2) adds them: this is a dot product, done for all pairs.
        gap = (offset * forward[:,None,:]).sum(axis =2) # + means j is ahead of i
        lateral = (offset * left[:, None, :]).sum(axis=2) # sideways distance

        # three numbers between 0 and 1, multiplied together:
        # 1) is car j ahead me? a smooth switch (0 behind, 1 ahead)
        is_ahead = 0.5 *(1.0 + np.tanh(gap / (2.0 * c.ahead_softness)))
        # 2) is it in my lane? 1 if straight ahead,falls with sideways distance
        same_lane = np.exp(-0.5 * (lateral / c.lane_sigma) ** 2)
        # 3) is it close? 1 if touching, falls with distance
        is_near = np.exp(-0.5 * (gap / c.range_m) ** 2)

        W = is_ahead * same_lane * is_near
        np.fill_diagonal(W, 0.0)
        return W , gap


    def neighbour_influence(self, X):
        """Acceleration each car gets from the cars around it, shape (N,).
        
                For every car i:   I_i = sum over j of  W[i, j] * phi[i, j]
                phi is a simple car-following reaction:
                  - if the car ahead is faster than me, speed up (and the opposite)
                  - if the gap is bigger than I want, speed up (and the opposite)
         """

        c = self.cpl
        W, gap = self.influence_weights(X)
        v = X[:, IDX_VX]  #  forward speed for each car

        # The gap each driver wants: a fixed part plus more when going faster
        desired_gap = c.standstill_gap + c.time_headway * v #shape (N,)

        # phi[i, j]: reaction of driver i to car j.
        # v[None, :] - v[:, None] is a table with  v_j - v_i  in position [i, j].
        # desired_gap[:, None] turns the list into a column, one value per row i.
        phi = (c.gain_speed * (v[None,:] - v[:, None]) 
               + c.gain_gap * (gap - desired_gap[:, None]))

        # Weight eeach reaction by W and add over j (across each row).
        return (W * phi).sum(axis=1)




    #Part 7: Later B, the vehicle transition


    def coupling_accel(self, G_next, X_prev):
        #Extra forward acceleration the DBN gives each car, shape (N,).
        
        # G_next : the NEW global state (time t)
        # X_prev : the car states BEFORE the step (time t-1)

        c = self.cpl

        #The two values of G the carss react to.

        mean_speed = G_next[IDX_MEAN_SPEED]
        congestion = G_next[IDX_CONGESTION]

        # 1) Macro -> micro: each car is pulled toward the mean speed of the
        # traffic. the pull is stronger when the road is congested.

        from_macro = c.gain_macro * (1.0 + congestion) * (mean_speed - X_prev[:, IDX_VX])

        # 2) Micro -> micro: the push from the cars around (Part 6)
        from_neighbours = self.neighbour_influence(X_prev)

        #add both then tanh is a smooth S_curve: so small values pass unchanged and big values are squeezed toward +-1
        # so result never exceed a_max
        return c.a_max * np.tanh((from_macro + from_neighbours) / c.a_max)


    def ssm_part(self, z, U):
        # Each car's OWN prediction from our SSM: x_next = f(x, u).
        # z: joint vector [G | car 1 | car 2 | ...]
        # U: driveer inputs, shape (N, 3) = (pedal, brake, drive ratio) per car 
        # returns a joint vector of the same size; the G part is zeros 
        #because G has no SSM of its own.

        _,X = self.unpack(z)

        #zip walks through cars, states and inputs together.
        #v.f(x, u) is the one_step prediction of your VehicleSSM.
        X_own = np.stack([v.f(x, u) for v, x, u in zip(self.vehicles, X, U)])

        return self.pack(np.zeros(N_GLOBAL), X_own)


    def coupling_part(self, z):
        # Eveerything the DBN add on top of the SSMs.
        # It contains NEW G, and a small speed change for each car..

        G, X = self.unpack(z)

        G_next = self.global_step(G, X) #Layer A (Part 5)

        # A table of Zeros, same shape as X (N, 11). Only the vx column gets 
        # a value: acceleration x time step = change of speed.

        push = np.zeros_like(X)
        push[:, IDX_VX] = self.dt * self.coupling_accel(G_next, X)

        return self.pack(G_next, push)


    def transition(self, z, U):
        # One full step of the whole network: z(t-1) -> z(t) (without noise)
        return self.ssm_part(z, U) + self.coupling_part(z)



    # part 8: Layer C (sensors) and the simulation

    def observe(self, X, U):
        #what the sensor of every car read, WITHOUT noise. Shape (n, 13).
        # X : car state (N, 11)
        # U : driver inputs (N, 3)
        # v.h(x,u) is the sensor model of our VehicleSSM

        return np.stack([v.h(x, u) for v, x, u in zip(self.vehicles, X, U)])


    def simulate(self, X0, U, steer=None, G0=None, rng=None, noise=True):
        #Run the whole network forward in time and record everything.
        """
        X0    : starting states of the cars, shape (N, 11)
        U     : driver inputs for every step, shape (T, N, 3)
        steer : optional steering angles, shape (T, N). If given, they REPLACE
                the model's own steering state at every step (otherwise the
                steering wanders randomly, which is rarely what you want).
        G0    : starting global state (if None, computed from the cars)
        rng   : random number generator (if None, a fixed seed is used)
        noise : False = noise-free run
        
        returns Gs (T, 4), Xs (T, N, 11), Ys (T, N, 13)
        """

        # A random number generator. With the same seed we get the same
        # "random" numbers every run, so our results can be repeated.

        rng = rng or np.random.default_rng(0)

        T = len(U)    # number of time steps
        scale = 1.0 if noise else 0.0    # 0 switches the noise off 

        # Standard deviation of each noise, taken from the diagonals of Q and R.
        process_std = np.sqrt(np.diag(self.Q)) # (dim, )
        obs_std = np.sqrt(np.diag(self.R)).reshape(self.N, N_OBS)   # (N, 13)

        # The starting joint vector 
        G_start = self.macro_summary(X0) if G0 is None else G0
        z = self.pack(G_start, X0)

        # Empty tables to fill in: time is the first axis.
        Gs = np.zeros((T, N_GLOBAL))
        Xs = np.zeros((T, self.N, N_STATE))
        Ys = np.zeros((T, self.N, N_OBS))

        for k in range(T):
            G, X = self.unpack(z)

            # Replace the steering angle by the command we want.
            if steer is not None:
                X = X.copy()   # do not modify z by accident 
                X[:, IDX_DELTA] = steer[k]
                z = self.pack(G, X)

            # Record the truth at time k.
            Gs[k], Xs[k] = G, X 

            # Sensors: the clean reading plus random noise.
            Ys[k] = self.observe(X, U[k]) + scale * rng.normal(size=(self.N, N_OBS)) 

            # Move one step forward, plus random process noise.
            z = self.transition(z, U[k]) + scale * process_std * rng.normal(size=self.dim)

        return Gs, Xs, Ys  

    # Part 9: INFERENCE, the joint Extended Kalman Filter 

    def transition_jacobian(self, z, U):
        #F: how the next step changes when z changes slightly.Shape (dim, dim).
        # Coupling part: slopes found numerically by nudging each number.
        #VehicalSSM._jac is the helper 
        F = vehicle._jac(self.coupling_part, z)

        #SSM part: add the exact slopes (matrix A) of each car on its own
        #diagonal block. Cars do not depend on other cars through the SSM.
        _. X = self.unpack(z)
        for i, vehicle in enumerate(self.vehicles):
            block = self.state_slice(i)
            F[block, block] += vehicle.matrices(X[i], U[i])["A"]
        return F


    def obs_jacobian(self, X, U):
        # H:how the sensor readings change when z changes. Shape (N*13, dim)
        H = np.zeros((self.N * N_OBS, self.dim))
        for i, vehicle in enumerate(self.dim):
            rows = slice(i * N_OBS, (i + 1) * N_OBS) # car i's sensors
            # Exact slopes (matrix C) of the SSM. The G columns stay Zero:
            # no sensor reads G directly.
            H[rows, self.state_slice(i)] = vehicle.matrices(X[i], U[i], discrete=False)["C"]
        return H  


  #
  #
  #
# tkekooon




















    
    def _update(self, z, P, U, y):
        """Correct the estimate with the sensor readings y (all cars, flattened).

        A missing reading is NaN and is simply ignored.
        The leading underscore means: internal helper, used only by filter().
        """
        mask = ~np.isnan(y)                    # True where a reading exists
        if not mask.any():                     # no data at all: nothing to do
            return z, P

        _, X = self.unpack(z)

        # Residual = what the sensors say minus what the model expected.
        residual = np.where(mask, y - self.observe(X, U).ravel(), 0.0)
        # The GPS course is an angle: wrap its error into -pi .. pi.
        residual[self.course_rows] = ssm.wrap(residual[self.course_rows])

        # Keep only the rows of the readings that exist.
        H = self.obs_jacobian(X, U)[mask]
        R = self.R[np.ix_(mask, mask)]
        residual = residual[mask]

        S = H @ P @ H.T + R                          # uncertainty of the residual
        K = np.linalg.solve(S, H @ P).T              # Kalman gain: trust in sensors
        I_KH = np.eye(self.dim) - K @ H

        z_new = z + K @ residual                      # corrected estimate
        P_new = I_KH @ P @ I_KH.T + K @ R @ K.T       # smaller uncertainty (Joseph form)
        return z_new, P_new

    def _predict(self, z, P, U):
        """Push the estimate one step forward with the model."""
        F = self.transition_jacobian(z, U)
        z_new = self.transition(z, U)
        P_new = F @ P @ F.T + self.Q                  # uncertainty grows
        return z_new, P_new

    def filter(self, U, Y, z0, P0):
        """Estimate G and all cars from the sensors only.

        U  : inputs (T, N, 3)
        Y  : sensor readings (T, N, 13); NaN = missing
        z0 : starting guess, vector of length dim
        P0 : uncertainty of the starting guess, (dim, dim)
        returns the estimates (T, dim) and their uncertainties (T, dim, dim)
        """
        T = len(Y)
        Y_flat = Y.reshape(T, -1)                     # one row of readings per step

        zs = np.zeros((T, self.dim))
        Ps = np.zeros((T, self.dim, self.dim))

        z, P = z0.copy(), P0.copy()
        for k in range(T):
            z, P = self._update(z, P, U[k], Y_flat[k])     # 1) use the sensors
            zs[k], Ps[k] = z, P                            # 2) record the estimate
            if k < T - 1:
                z, P = self._predict(z, P, U[k])           # 3) step forward
        return zs, Ps
