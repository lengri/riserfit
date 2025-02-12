import numpy as np
import pickle
import os, sys 
import matplotlib.pyplot as plt
testdir = os.path.dirname(__file__)
srcdir = '../'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
import riserfit as rf
os.chdir("C:/Users/Lennart/lennartGit/personal/riserfit")

# For this to work, we will need one k value per riser profile
# And calculate k per profile.

def load_StatsMC_from_stub(stub, type="StatsMC"):
    try:
        id = f"SC_{stub}_{type}_instance.gz"
        file = rf.load_instance(f"\\Data\\StatsMC\\StatsMC_Pat\\{id}")
    except:
        id = f"SH_{stub}_{type}_instance.gz"
        file = rf.load_instance(f"\\Data\\StatsMC\\StatsMC_Pat\\{id}")
    return file

def load_Riser_from_stub(stub):
    try:
        return rf.load_instance(f"\\Data\\Risers\\risers_pat\\SH_{stub}_Riser_instance.gz")
    except:
        return rf.load_instance(f"\\Data\\Risers\\risers_pat\\SC_{stub}_Riser_instance.gz")


def k_per_profile(instance, smc):
    
    n = len(instance.name)
    k = np.zeros(n)
    x = np.zeros(n)
    
    # set the cn age and cn age uncertainty of the instance using smc info.
    t = [smc.t for _ in range(0, n)]
    
    t_sigma = [smc.t_sigma for _ in range(0, n)]
    instance.cn_age, instance.cn_age_sigma = t, t_sigma
    
    if not np.isnan(t[0][0]):
        print(instance.name[0], t[0], t_sigma[0])
        for i, _ in enumerate(instance.name):
            r = instance.extract_subset([i])
            smc = r.construct_StatsMC_instance(
                kt_parameter_name="nonlin_best_t",
                kt_lb_parameter_name="nonlin_lower_t",
                kt_ub_parameter_name="nonlin_upper_t"
            )[0]
            #print("hi1")
            smc.construct_kt_kde(max_val=1e4)
            #print("hi2")
            smc.construct_initial_t_kde(t_resolution=0.1)
            #print("hi3")
            smc.construct_MC_k_kde(n=10_000, k_resolution=0.1)
            k[i] = smc.k_kde.inverse_cdf(0.5)
            x[i] = r.x[0][0]
    
        return (x, k)
    else:
        return ([], [])

if __name__ == "__main__":
    
    dGray = "#3C3C3C"  
    os.chdir(r"C:\Users\Lennart\OneDrive\Desktop\GFZ\Morphological dating")    
    
    # SC terraces
    SC_terraces = [
            "CA", "VI", "MQ", "NU", "SF", 
            "LA", "MA", "PB", "OR"]
    
    color_dict = {
        "EA": '#7D4901', "CA": '#782E6F',
        "VI": '#B974B3', "IN": '#B2DF8A',
        "LU": '#33a02c', "MQ": '#33A02C',
        "NU": '#315C2C', "PPI": '#2C67A9',
        "PPII": "#92b8c9",
        "SF": '#E31A1C', "LA": '#434499',
        "MA": '#7EBDB8', "PB": '#35625D',
        "OR": '#555555', "BI": "#c5b4d4",
        "LL": "#ff7f00", "PV": "#f5c06e",
        "C2": "#b974b3", "C1": "#782e6f",
        "Y1": "#315c2c", "BII": "#643f96",
        "UB": "#fccd30", "OR": "#969696",
        "CT": "#555555", "??": "#4d4ca7",
        "LR": "#eeeb66"
    }

    risers = [load_Riser_from_stub(t) for t in SC_terraces]
    smc = [load_StatsMC_from_stub(t) for t in SC_terraces]

    # get k values...
    
    fg, ax = plt.subplots(4, 1, figsize=(7.5,6))
    for i, (r, s) in enumerate(zip(risers, smc)):
        x, k = k_per_profile(r,s)
        ax[1].scatter(x, k, s=12, ec=dGray, lw=0.4, fc=color_dict[SC_terraces[i]])
    ax[1].set_ylim(0, 10)
    plt.show()