import matryoshka.emulator as MatEmu
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from scipy.stats import norm
import zeus
from scipy.interpolate import interp1d
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--redshift', help='Redshift of mock to analyse. Can be 0.38, 0.51, 0.61',
                    required=True)
parser.add_argument('--volume', help='Volume of mock to analyse. Can be 1000, 2000, 3000, 3700, 4000, or 5000',
                    required=True)
parser.add_argument('--repo_path', help='Path to project repo.', required=True)
parser.add_argument('--save_path', help='Path to save results.', required=True)
parser.add_argument("--kmax", help="the maximum k-value to be used in the fit.", 
                    default=None)
parser.add_argument("--kmin", help="the maximum k-value to be used in the fit.", 
                    default=None)
parser.add_argument("--chain_fname", help="file name for saved chain.",
                    default="chain--EFTEMU_z-{a}_kmin-{b}_kmax-{c}.npy")
args = parser.parse_args()

# Check repo path.
path_to_repo = str(args.repo_path)
if not os.path.isdir(path_to_repo):
    raise ValueError("Repo not found at specified path.")
else:
    if path_to_repo[-1] is not "/":
        path_to_repo += "/"

# Check save path.
save_dir = str(args.save_path)
if not os.path.isdir(save_dir):
    raise ValueError("Save directory not found.")
else:
    if save_dir[-1] is not "/":
        save_dir += "/"

# Define the underlying true cosmology.
cosmo_true = np.array([0.11933, 0.02242, 0.6766, 3.047, 0.9665])
ng = 3e-4
fb_true = cosmo_true[1]/(cosmo_true[0]+cosmo_true[1])

# Check input redshift and define relelvant true bias parameters.
redshift = float(args.redshift)
if redshift == 0.38:
    bs_true = np.array([1.73, 1.0, -1.0, 0.0, 0.2, -10.03, 0., 0., 0., -2.1])
elif redshift == 0.51 or redshift == 0.61:
    bs_true = np.array([2.22, 1.2, 0.1, 0.0, 0.4, -7.7, 0., 0., 0., -3.7])
else:
    raise ValueError("Please specify a relevant redshift. Can be 0.38, 0.51, or 0.61")

# Check volume.
volume = int(args.volume)
if volume not in [1000,2000,3000,3700,4000,5000]:
    raise ValueError("Please specify a relevant volume. Can be 1000, 2000, 3000, 3700, 4000, or 5000.")

# Load the desired mock data.
P0_true = np.load(path_to_repo+"data/P18/z{z}/poles/P0_P18--z-{z}_optiresum-False.npy".format(z=redshift))[1]
print("Loaded: P0_P18--z-{z}_optiresum-False.npy".format(z=redshift))
P2_true = np.load(path_to_repo+"data/P18/z{z}/poles/P2_P18--z-{z}_optiresum-False.npy".format(z=redshift))[1]
print("Loaded: P2_P18--z-{z}_optiresum-False.npy".format(z=redshift))
klin = np.load(path_to_repo+"data/P18/z{z}/poles/P2_P18--z-{z}_optiresum-False.npy".format(z=redshift))[0]
cov = np.load(path_to_repo+"data/P18/z{z}/covs/cov_P18--z-{z}_Vs-{V}.npy".format(z=redshift, V=volume))
print("Loaded: cov_P18--z-{z}_Vs-{V}.npy".format(z=redshift, V=volume))
icov = np.linalg.inv(cov)

# Impose kmin
if args.kmin is not None:
    kmin = float(args.kmin)
    if kmin<MatEmu.kbird[:39].min():
        raise ValueError("kmin value not valid.")
    else:
        ks_good = klin>kmin
        klin = klin[ks_good]
        P0_true = P0_true[ks_good]
        P2_true = P2_true[ks_good]
else:
    kmin='def'

# Impose kmax
if args.kmax is not None:
    kmax = float(args.kmin)
    if kmax>MatEmu.kbird[:39].max():
        raise ValueError("kmin value not valid.")
    else:
        ks_good = klin<kmax
        klin = klin[ks_good]
        P0_true = P0_true[ks_good]
        P2_true = P2_true[ks_good]
else:
    kmax='def'

# Initalise emulators.
print("Initalising emulators...")
P0_emu = MatEmu.EFT(multipole=0, version='EFTv2', redshift=redshift)
print("P0 done.")
P2_emu = MatEmu.EFT(multipole=2, version='EFTv2', redshift=redshift)
print("P2 done.")

########### MCMC FUNCS ##################################

def log_prior(theta, ng, cosmo_bounds):
    # Oc, Ob, h, As, ns
    cosmo = theta[:,:5]
    # b1, c2, b3, c4, cct, cr1, cr2
    bias = theta[:,5:12]
    # ce1, cmono, cquad
    stoch = theta[:,12:]
    
    # Evaluate uniform prior on cosmo params.
    box_cosmo = np.greater(cosmo[:, 0], cosmo_bounds[0,0]) & np.less(cosmo[:, 0], cosmo_bounds[0,1]) \
        & np.greater(cosmo[:, 1], cosmo_bounds[1,0]) & np.less(cosmo[:, 1], cosmo_bounds[1,1]) \
        & np.greater(cosmo[:, 2], cosmo_bounds[2,0]) & np.less(cosmo[:, 2], cosmo_bounds[2,1])\
        & np.greater(cosmo[:, 3], cosmo_bounds[3,0]) & np.less(cosmo[:, 3], cosmo_bounds[3,1])\
        & np.greater(cosmo[:, 4], cosmo_bounds[4,0]) & np.less(cosmo[:, 4], cosmo_bounds[4,1])
    
    box_bias = np.greater(bias[:, 0], 0.) & np.less(bias[:, 0], 4.) \
        & np.greater(bias[:, 1], -4.) & np.less(bias[:, 1], 4.)
    
    return np.where(box_cosmo, 0, -np.inf)+np.where(box_bias, 0, -np.inf)\
           +norm.logpdf(bias[:,2], loc=0, scale=2)\
           +norm.logpdf(bias[:,3], loc=0, scale=2)\
           +norm.logpdf(bias[:,4], loc=0, scale=2)\
           +norm.logpdf(bias[:,5], loc=0, scale=8)\
           +norm.logpdf(bias[:,6], loc=0, scale=4)\
           +norm.logpdf(stoch[:,0]/ng, loc=0, scale=400)\
           +norm.logpdf(stoch[:,1], loc=0, scale=2)\
           +norm.logpdf(stoch[:,2], loc=0, scale=2)

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

def log_prob(theta, kobs, obs, icov, ng, cosmo_bounds, fb, fixed_vals):
    
    # Convert list to array
    theta = np.vstack(theta)
    
    # Fix parameters
    theta = fix_params(theta, fixed_vals, fb)
    
    # Evaluate prior
    lp = log_prior(theta, ng, cosmo_bounds)
    
    # Evaluate likelihood
    ll = log_like(theta, kobs, obs, icov)
    
    return ll+lp

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

def neg_log_prob(theta, kobs, obs, icov, ng, cosmo_bounds, fb, fixed_vals):
    theta = np.array(theta).reshape(1, -1)
    return -(log_prob(theta, kobs, obs, icov, ng, cosmo_bounds, fb, fixed_vals))

def c_to_b(c2, c4):
    return np.array([1/np.sqrt(2)*(c2+c4),1/np.sqrt(2)*(c2-c4)])

#############################################

# Define number of dimensions and walkers for the MCMC
ndim=10
nwalk=2*ndim

# Generate some random inital guess
init_guess = np.concatenate([cosmo_true[[0,2,3]], bs_true[[0,1,2,4,5,7,9]]])\
             +1e-3*np.random.randn(ndim, )

# Define cosmological prior bounds based on the training data,
emu_bounds = np.stack([P0_emu.P11.scalers[0].min_val,
                       P0_emu.P11.scalers[0].min_val+P0_emu.P11.scalers[0].diff]).T

# Define the values of the fixed parameters ns, c4, cr2, cm.
fixed_params = np.array([cosmo_true[-1], 0., 0., 0.])

# Find MAP
print("Finding MAP...")
results = minimize(neg_log_prob, init_guess, method='Nelder-Mead', options={'disp': True, "maxiter": 100000},
                   args=(klin, np.concatenate([P0_true, P2_true]), icov, ng, emu_bounds, fb_true, fixed_params))

# Check if previous results exist.
MAPest_fname = "MAP--EFTEMU_z-{z}_V-{V}_kmin-{k1}_kmax-{k2}_0.npy".format(z=redshift, V=volume, k1=kmin, k2=kmax)
if os.path.isfile(save_dir+MAPest_fname):
    i=0
    while os.path.isfile(save_dir+MAPest_fname):
        i += 1
        MAPest_fname = "MAP--EFTEMU_z-{z}_V-{V}_kmin-{k1}_kmax-{k2}_{i}.npy"\
                       .format(z=redshift, V=volume, k1=kmin, k2=kmax, i=i)

# Save MAP estimate.
np.save(save_dir+MAPest_fname, results.x)
print("MAP saved: {fname}".format(fname=MAPest_fname))

# Initalise random positions for walkers
init_pos = results.x+1e-3*np.random.randn(nwalk, ndim)

# Define convergence callbacks.
cb0 = zeus.callbacks.AutocorrelationCallback(ncheck=200, dact=0.01, nact=100, discard=0.5)
cb1 = zeus.callbacks.MinIterCallback(nmin=5000)

# Run MCMC
sampler = zeus.EnsembleSampler(nwalk, ndim, log_prob, vectorize=True, 
                               args=(klin, np.concatenate([P0_true, P2_true]),
                                     icov, ng, emu_bounds, fb_true, fixed_params), 
                               maxiter=10**5)
sampler.run_mcmc(init_pos, 50000, progress=True, callbacks=[cb0, cb1])

# Discard burn-in and thin chain.
taus = cb0.estimates
flat_chain = sampler.get_chain(discard=5*int(taus[-1]), flat=True, thin=int(taus[-1]))

# Check if previous results exist.
chain_fname = "chain--EFTEMU_z-{z}_V-{V}_kmin-{k1}_kmax-{k2}_0.npy".format(z=redshift, V=volume, k1=kmin, k2=kmax)
if os.path.isfile(save_dir+chain_fname):
    i=0
    while os.path.isfile(save_dir+chain_fname):
        i += 1
        MAPest_fname = "chain--EFTEMU_z-{z}_V-{V}_kmin-{k1}_kmax-{k2}_{i}.npy"\
                       .format(z=redshift, V=volume, k1=kmin, k2=kmax, i=i)

# Save chain.
np.save(save_dir+chain_fname, flat_chain)
print("MAP saved: {fname}".format(fname=chain_fname))