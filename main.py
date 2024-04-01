import numpy as np 
import src 
import sys 
import os 
import wandb 

M = int(sys.argv[1]) 
T = float(sys.argv[2])
n_samples = int(sys.argv[3])


config = {
    "size_of_lattice": M, 
    "temperature": T, 
    "n_samples": n_samples, 
    "project": f"ising-model-size={M}", 
    "entity": "jorneirikbetten-complexity"
}



wandb.init(
    entity=config["entity"],
    project=config["project"],
    tags=["MCMC", "Ising Model"], 
    name=f'n_samples_{n_samples}_size_{M}_at_temp_{T:.1f}',
    config=config,
)


rng = np.random.default_rng(123) 
grid = src.NumpyGrid(M)
sampler = src.NumpyMC(grid, n_samples, T, rng, warmup=True)

df = sampler.sample()
data_path = os.getcwd() + "/results/" + f"lattice_{M}/"
if not os.path.exists(data_path): 
    os.makedirs(data_path)
df.to_csv(data_path + f"temperature_{T:.1f}.csv")