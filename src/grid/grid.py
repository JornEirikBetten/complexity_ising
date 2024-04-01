import numpy as np 
"""
    The Ising Model: 
        grid = MxM lattice sites for spins. 
        s_k = {+1, -1} (spin can be up or down)
        s = {s_k} k\in{1, 2, \dots, M^2} (spin configuration)

        Energy, Hamiltonian of system: 
        H(s) = -\sum_{i}\sum_{j\in{N_i}}J_{ij}s_is_j - mu\sum_{j}h_js_j
            where J_{ij} is the interaction energy, and N_i is the immediate neighborhood of i, 
            h_j is an external magnetic field, and mu is the magnetic moment of the spins. 
            
        Magnetic moment: 
            An analogy of the magnetic moment is the lever arm in the definition of torque, 
            
                tau_ = r_ \cross F_
            because the pseudo-vector torque of the magnetic field is 
            
                tau_mag = m_ \cross B_, 
            where B_ is the magnetic field, and m_ is the magnetic torque.  

"""


class NumpyGrid: 
    def __init__(self, M):
        self.M = M 
        self.grid = np.ones((M+2, M+2), dtype=np.int32)
        self._J = 0.5
        print(f"Initialized an {self.M} times {self.M} iso-spin lattice with positive spins.")
        idxs, down, right = self.find_neighbors_and_indices() 
        self.idxs = idxs 
        self.down = down 
        self.right = right 
        
        
    def find_neighbors_and_indices(self):
        idx = np.indices((self.M, self.M), sparse=False) + 1
        i_idxs = np.reshape(idx[0], -1)
        j_idxs = np.reshape(idx[1], -1)
        idxs = tuple((i_idxs, j_idxs)) # All lattice sites of the interior
        
        # Define all right and down neighbors
        down = tuple((idxs[0] + 1, idxs[1]))
        right = tuple((idxs[0], idxs[1] + 1))
        
        return idxs, down, right 
        
    def calculate_energy_proposal(self, proposal): 
        pairwise_interactions = - self._J * proposal[self.idxs] * proposal[self.down] - self._J * proposal[self.idxs] * proposal[self.down]
        return np.sum(pairwise_interactions)
        
    def calculate_energy(self): 
        """Calculate potential energy of the lattice given Hamiltionian:
            H(s) = -\sum_{ij} J_{ij}s_is_j 
           where B_ = 0_ is assumed. 
        """ 
        pairwise_interactions = -self._J * self.grid[self.idxs] * self.grid[self.down] - self._J * self.grid[self.idxs] * self.grid[self.right] 
        return np.sum(pairwise_interactions)
        
    def calculate_magnetization(self): 
        return np.sum(self.grid[self.idxs])/(self.M*self.M)
    
    def flip_spin(self, idx): 
        # Flip spin 
        self.grid[idx] *= -1 
        # Copy outer grid for periodic boundary conditions 
        self.grid[0, :] = self.grid[self.M, :].copy()
        self.grid[:, 0] = self.grid[:, self.M].copy()
        self.grid[self.M+1, :] = self.grid[1, :].copy()
        self.grid[:, self.M+1] = self.grid[:, 1].copy() 
        return 0 
    
    def propose_flip(self, idx): 
        # Flip spin 
        proposal = self.grid.copy()
        original_energy = self.calculate_energy_proposal(proposal) 
        proposal[idx] *= -1 
        proposal_energy = self.calculate_energy_proposal(proposal)
        return proposal_energy - original_energy, proposal 
        
    
    
if __name__ == "__main__": 
    grid = NumpyGrid(5)
    print(f"Energy={grid.calculate_energy()}")
    print(f"Magnetization={grid.calculate_magnetization()}")