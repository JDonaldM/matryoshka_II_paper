import numpy as np
import matryoshka.eft_funcs as MatEFT
from scipy.interpolate import interp1d
from classy import Class
import pybird
import argparse
import glob

path_to_repo = "/mnt/lustre/jdonaldm/matryoshka_II/"

parser = argparse.ArgumentParser()
parser.add_argument('--redshift', help='Redshift of mock to analyse. Can be 0.38, 0.51, 0.61',
                    required=True)
parser.add_argument('--volume', help='Volume of mock to analyse. Can be 1000, 2000, 3000, 3700, 4000, or 5000',
                    required=True)
parser.add_argument('--chunk', help='Specify the chunk ID. This will correspond to 10000 samples from the emulator chain',
                    required=True)
parser.add_argument('--chain_vol', help='Specify the chain to draw samples from.',
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

P0 = np.load(path_to_repo+f"data/P18/z{args.redshift}/poles/P0_P18--z-{args.redshift}_optiresum-False.npy")[1]
P2 = np.load(path_to_repo+f"data/P18/z{args.redshift}/poles/P2_P18--z-{args.redshift}_optiresum-False.npy")[1]
klin = np.load(path_to_repo+f"data/P18/z{args.redshift}/poles/P2_P18--z-{args.redshift}_optiresum-False.npy")[0]

# Check volume.
volume = int(args.volume)
if volume not in [1000,2000,3000,3700,4000,5000]:
    raise ValueError("Please specify a relevant volume. Can be 1000, 2000, 3000, 3700, 4000, or 5000.")

def log_like(theta, ks, obs, icov, ng):
    
    res = np.concatenate(power_pred(theta, ng, ks))-obs

    return -0.5*np.dot(np.dot(res,icov),res.T)

def power_pred(theta, ng, ks):
    cosmo = theta[:5] # Oc, Ob, h, As, ns
    bias = theta[5:12] # b1, c2, b3, c4, cct, cr1, cr2
    stoch = theta[12:] # ce1, cmono, cquad
        
    c2 = np.copy(bias[1])
    c4 = np.copy(bias[3])
    
    bias[1] = (c2+c4)/np.sqrt(2)
    bias[3] = (c2-c4)/np.sqrt(2)
    
    M.set({'ln10^{10}A_s': cosmo[3],
           'n_s': cosmo[4],
           'h': cosmo[2],
           'omega_b': cosmo[1],
           'omega_cdm': cosmo[0],
          })
    
    # Calculate the linear power spectrum.
    M.compute()
        
    # Convert to (Mpc/h)^3.
    Pk = [M.pk(ki*M.h(), float(args.redshift))*M.h()**3 for ki in kk]
    
    f = M.scale_independent_growth_factor_f(float(args.redshift))
    
    bird = pybird.Bird(kk, Pk, f, z=float(args.redshift), which='all', co=common)

    # Calculate the desired functions.
    nonlinear.PsCf(bird)
    bird.setPsCfl()
    resum.Ps(bird)
    
    # Compute multipoles from Pn.
    P0_pred = MatEFT.multipole([bird.P11l[0][:,:39], bird.Ploopl[0][:,:39], bird.Pctl[0][:,:39]], 
                               bias, f, stochastic=stoch, ng=ng, multipole=0,
                               kbins=pybird.kbird[:39])
 
    P2_pred = MatEFT.multipole([bird.P11l[1][:,:39], bird.Ploopl[1][:,:39], bird.Pctl[1][:,:39]], 
                               bias, f, stochastic=stoch, ng=ng, multipole=2,
                               kbins=pybird.kbird[:39])
    
    return interp1d(pybird.kbird[:39], P0_pred)(ks), interp1d(pybird.kbird[:39], P2_pred)(ks)

def fix_params(theta, fix_val, fb):
    var_id = np.array([0,2,3,5,6,7,9,10,12,14])
    fix_id = np.array([4,8,11,13])
    fix_theta = np.zeros((15, ))
    fix_theta[var_id] = theta
    fix_theta[fix_id] = fix_val
    fix_theta[1] = -fb*theta[0]/(fb-1)
    return fix_theta

M = Class()
M.set({'output': 'mPk',
       'P_k_max_1/Mpc': 1.0,
       'z_max_pk': float(args.redshift)})
common = pybird.Common(optiresum=False)
nonlinear = pybird.NonLinear(load=True, save=True, co=common)
resum = pybird.Resum(co=common)
kk = np.logspace(-5, 0, 200)

# Load cov to be used when computing PyBird likes.
cov_file = glob.glob(path_to_repo+f"data/P18/z{args.redshift}/covs/cov_P18--z-{args.redshift}_Vs-{args.volume}*.npy")[0]
print(cov_file)
icov = np.linalg.inv(np.load(cov_file))
i = int(args.chunk)
save_fname = f"results/pybird_likes--EFTEMU_z-{args.redshift}_Vc-{int(args.chain_vol)}_Vi-{args.volume}_kmin-def_kmax-def_{i}.npy"
chain_file = glob.glob(path_to_repo+f"results/chain--EFTEMU_z-{args.redshift}_V-{int(args.chain_vol)}*--shuffled.npy")[0]
print(chain_file)
if i == 0:
    emu_chain = np.load(chain_file)[-10000:]
else:
    emu_chain = np.load(chain_file)[-10000*(i+1):-10000*i]

# Loop over chain samples and compute likelihood
likes = np.zeros((emu_chain.shape[0]))
#for i in tqdm(range(emu_chain.shape[0])):
for i in range(emu_chain.shape[0]):
    likes[i] = log_like(fix_params(emu_chain[i], np.array([cosmo_true[-1],0.,0.,0.]), fb_true),
                                    klin, np.concatenate([P0,P2]), icov, ng)

np.save(path_to_repo+save_fname, likes)