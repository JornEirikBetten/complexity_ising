import numpy as np 
import wandb 
import pandas as pd 
import tqdm 

class NumpyMC: 
    def __init__(self, grid, n_samples, T, rng, warmup=False, warmup_steps=1_000_000): 
        self.grid = grid.grid 
        self.idxs, self.down, self.right = grid.idxs, grid.down, grid.right 
        self.M = grid.grid.shape[0] - 2
        self.idx_range = np.arange(1, self.M + 1, 1)
        self.n_samples = n_samples 
        self.T = T # Temperature 
        self._J = grid._J
        self._rng = rng 
        self.warmup = warmup
        self.warmup_steps = warmup_steps
        self.energy = grid.calculate_energy() 
        self.magnetization = grid.calculate_magnetization() 
        self.step = 0
        self.accepted = 0 
        
    def warmup_step(self): 
        proposal_flip = tuple((self._rng.choice(self.idx_range, size=(1, )), self._rng.choice(self.idx_range, size=(1, ))))
        dE, proposal_grid = self.propose_flip(proposal_flip)
        random_draw = self._rng.random() 
        log_r = np.log(random_draw)
        if dE <= 0: 
            accept = True 
            log_prob = 0
        else: 
            log_prob = -np.log(dE) + np.log(self.T)
            accept = log_prob > log_r 
        if accept: 
            self.grid = proposal_grid 
        return 0 
        
    def metropolis_step(self, statistics): 
        proposal_flip = tuple((self._rng.choice(self.idx_range, size=(1, )), self._rng.choice(self.idx_range, size=(1, ))))
        dE, proposal_grid = self.propose_flip(proposal_flip)
        random_draw = self._rng.random() 
        log_r = np.log(random_draw)
        if dE <= 0: 
            accept = True 
            log_prob = 0
        else: 
            log_prob = -np.log(dE) + np.log(self.T)
            accept = log_prob > log_r 
        self.accepted += accept
        if accept: 
            self.grid = proposal_grid
            self.energy += dE 
            self.magnetization = self.calculate_magnetization(proposal_grid)
        
        statistics["energy"].append(self.energy)
        statistics["magnetization"].append(self.magnetization)
        self.step += 1 
         
        metrics = {"energy": self.energy, 
                   "mean_energy": np.mean(statistics["energy"]), 
                   "magnetization": self.magnetization, 
                   "mean_magnetization": np.mean(statistics["magnetization"]),  
                   "timestep": self.step+1, 
                   "accepted": self.accepted, 
                   "log_prob": log_prob, 
                   "log_r": log_r, 
                   } 
        if self.step % 100 == 0: 
            wandb.log(metrics)
        return 0 
    
    def sample(self): 
        statistics = {"energy": [], 
                      "magnetization": [], 
                      "step": [i+1 for i in range(self.n_samples)]}
        if self.warmup: 
            print("Warming up")
            for i in tqdm.tqdm(range(self.warmup_steps)): 
                self.warmup_step()
        print("Sampling")
        for i in tqdm.tqdm(range(self.n_samples)): 
            self.metropolis_step(statistics)
            
        df = pd.DataFrame(data=statistics)
        
        return df  
    
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
    
    def calculate_magnetization(self, grid): 
        return np.sum(grid[self.idxs])/(self.M * self.M)
    
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