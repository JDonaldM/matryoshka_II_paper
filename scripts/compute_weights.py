import matryoshka.emulator as MatEmu
import numpy as np
from scipy.interpolate import interp1d
import glob
import argparse

path_to_repo = "/Users/jamie/Desktop/GitHubProjects/matryoshka_II_paper/"

parser = argparse.ArgumentParser()
parser.add_argument('--redshift', help='Redshift of mock to analyse. Can be 0.38, 0.51, 0.61',
                    required=True)
parser.add_argument('--volume_target', help='Volume of target distribution. Can be 1000, 2000, 3000, 3700, 4000, or 5000',
                    required=True)
parser.add_argument('--volume_proposal', help='Volume of proposal distribution. Can be 1000, 2000, 3000, 3700, 4000, or 5000',
                    required=True)
args = parser.parse_args()

# Define the underlying true cosmology.
cosmo_true = np.array([0.11933, 0.02242, 0.6766, 3.047, 0.9665])
ng_LOWZ = 4e-4
ng_CMASS = 4.5e-4
fb_true = cosmo_true[1]/(cosmo_true[0]+cosmo_true[1])

# Check input redshift and define relelvant true bias parameters.
redshift = float(args.redshift)
if redshift == 0.38:
    ng = ng_LOWZ
elif redshift == 0.51 or redshift == 0.61:
    ng = ng_CMASS
else:
    raise ValueError("Please specify a relevant redshift. Can be 0.38, 0.51, or 0.61")

P0_true = np.load(path_to_repo+f"data/P18/z{args.redshift}/poles/P0_P18--z-{args.redshift}_optiresum-False.npy")[1]
P2_true = np.load(path_to_repo+f"data/P18/z{args.redshift}/poles/P2_P18--z-{args.redshift}_optiresum-False.npy")[1]
klin = np.load(path_to_repo+f"data/P18/z{args.redshift}/poles/P2_P18--z-{args.redshift}_optiresum-False.npy")[0]

# Check volume.
volume_target = int(args.volume_target)
volume_proposal = int(args.volume_proposal)
if (volume_target not in [1000,2000,3000,4000,5000]) or (volume_proposal not in [1000,2000,3000,4000,5000]):
    raise ValueError("Please specify a relevant volume. Can be 1000, 2000, 3000, 3700, 4000, or 5000.")

P0_emu = MatEmu.EFT(multipole=0, redshift=redshift)
P2_emu = MatEmu.EFT(multipole=2, redshift=redshift)

# Functions for likelihood.
###############################################################

def log_like(theta, kobs, obs, icov):
    # Oc, Ob, h, As, ns
    cosmo = theta[:,:5]
    # b1, c2, b3, c4, cct, cr1, cr2
    bias = theta[:,5:12]
    # ce1, cmono, cquad
    stoch = theta[:,12:]
    
    
    c2 = np.copy(bias[:,1])
    c4 = np.copy(bias[:,3])
    
    bias[:,1] = (c2+c4)/np.sqrt(2)
    bias[:,3] = (c2-c4)/np.sqrt(2)
            
    P0_pred = P0_emu.emu_predict(cosmo, bias, stochastic=stoch, ng=ng)
    P2_pred = P2_emu.emu_predict(cosmo, bias, stochastic=stoch, ng=ng)
        
    preds = np.hstack([interp1d(MatEmu.kbird[:39], P0_pred)(kobs), 
                           interp1d(MatEmu.kbird[:39], P2_pred)(kobs)])
    
    res = preds-obs
    
    return -0.5*np.einsum("nj,ij,in->n", res, icov, res.T)

def fix_params(theta, fix_val, fb):
    
    # Define indicies of parameters that vary.
    var_id = np.array([0,2,3,5,6,7,9,10,12,14])
    
    # Define indicies of fixed params.
    fix_id = np.array([4,8,11,13])
    
    fix_theta = np.zeros((theta.shape[0], 15))
    fix_theta[:,var_id] = theta
    fix_theta[:,fix_id] = np.vstack(theta.shape[0]*[fix_val])
    
    # Comput w_b from baryon fraction and w_c
    fix_theta[:,1] = -fb*theta[:,0]/(fb-1)
    
    return fix_theta

###############################################################

fixed_params = np.array([cosmo_true[-1], 0., 0., 0.])

file_list = sorted(glob.glob(path_to_repo+f"results/pybird_likes*{redshift}*{volume_proposal}*{volume_target}*"))

# Check there is atleast one pybird likelihood file.
if len(file_list) < 1:
    raise ValueError(f"No PyBird likelihood file for z={redshift}, V_target={volume_target}, V_proposal={volume_proposal}.")

# Combine all PyBird likelihood files.
pybird_likes = []
for file_i in file_list[::-1]:
    pybird_likes.append(np.load(file_i))
pybird_likes = np.concatenate(pybird_likes)

# Load chain file calculated with emulator.
# Acts as samples from proposal.
chain_file = glob.glob(path_to_repo+f"results/chain--EFTEMU_z-{redshift}_V-{volume_proposal}*--shuffled.npy")[0]
emu_chain = np.load(chain_file)[-10000*len(file_list):]

# Load the covariance tht was used to calculate the chain.
cov_file = glob.glob(path_to_repo+f"data/P18/z{redshift}/covs/cov_P18--z-{redshift}_Vs-{volume_proposal}*.npy")[0]
icov = np.linalg.inv(np.load(cov_file))

# Fix the relevant parameters.
fixed_chain = fix_params(emu_chain, fixed_params, fb_true)

# Compute the proposal likelihoof.
EFTEMU_likes = log_like(fixed_chain, klin, np.concatenate([P0_true, P2_true]), icov)

# Compute the likelihood ratio.
ratio = np.exp(pybird_likes)/np.exp(EFTEMU_likes)

# Save weights file.
np.save(path_to_repo+f"results/weights--EFTEMU_z-{redshift}_Vc-{volume_proposal}_Vi-{volume_target}_kmin-def_kmax-def_all.npy", ratio)

# Compute the effective sample size.
ESS = np.sum(ratio)**2/np.sum(ratio**2)
print(ESS, ESS/emu_chain.shape[0])