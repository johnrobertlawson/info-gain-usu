"""
TODO:
    * Scorecard for variables like rho, windowing that show region in which
    the model does badly/well, but measured with both BS and IG.
    * The max filter then identifies periods of maximum activity of something
    like UH, a proxy for supercells. These are multiple collated storm days.
    Intermittent so they are already low prob, and they overlap in
    (?) ways within an ensemble - need a mutual information chart

"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as M
import scipy.ndimage
from scipy.ndimage import maximum_filter, maximum_filter1d, median_filter

from lorenz63 import Lorenz63Simulator

### SETTINGS ###
rng = np.random.default_rng(seed=27)


### FUNCTIONS ###
def reshape_and_visualize(data, title, shape):
    # Reshape the data into a 2D array for visualization
    reshaped_data = np.reshape(data, (shape,shape))

    # Plot the 2D array as a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(reshaped_data, cmap='Greys', interpolation='none')
    plt.title(title)
    plt.show()

def plot_time_series(data,ylims,pc=None,exceed=None):
    fig,ax = plt.subplots(1,figsize=(12,6))
    ax.plot(data)
    if exceed:
        qpc = exceed
    else:
        qpc = np.percentile(data, pc)
    ax.axhline(xmin=0,xmax=data.size+1,y=qpc)
    ax.set_ylim(ylims)
    fig.show()
    return

def do_exceedence(data,pc=None,exceed=None,do_plot=False):
    if exceed is None:
        qpc = np.percentile(data, pc)
    else:
        qpc = exceed
    exceed0 = data>qpc
    exceed = maximum_filter1d(exceed0,size=100,mode="constant",cval=0)
    if do_plot:
        fig,ax = plt.subplots(1,figsize=(12,6))
        ax.plot(exceed)
        fig.show()
    return exceed

# def plot_exceedence(data,pc=None,exceed=None):
#     """
#     Also has max filter to simulate time periods
#     """
#     if exceed is None:
#         qpc = np.percentile(data, pc)
#     else:
#         qpc = exceed
#     exceed0 = data>qpc
#     exceed = maximum_filter1d(exceed0,size=100,mode="constant",cval=0)
#     return exceed

# def plot_truth(truth_data,pc,ylims):
#     data = do_exceedence(truth_data,pc)


def plot_max_filter(data,ylims):
    # fig,ax = plt.subplots(1,figsize=(12,6))
    # max_data = maximum_filter(data,size=500,mode="constant",cval=data.mean())
    # ax.plot(max_data)
    # ax.set_ylim(ylims)
    # fig.show()
    # return max_data
    pass

def moving_average(data,n):
    cs = np.cumsum(data)
    cs[n:] = cs[n:] - cs[:-n]
    return cs[n-1:]/n

def plot_probs(data):
    # n is the prob smoothing
    fig,ax = plt.subplots(1,figsize=(10,8))
    ax.plot(data)
    ax.set_ylim([0,1])
    fig.show()
    return

def plot_truth(data,n=None):
    fig,ax = plt.subplots(1,figsize=(8,4))
    ax.imshow(data[np.newaxis,:],cmap=M.cm.binary,aspect="auto")
    fig.show()
    return

def do_smooth(probs,n):
    return median_filter(probs,size=n)

def main():
    # ylims = [-50,50]
    # Might rename shape to "day length"
    daylength = 5000
    ylims = [40,250]
    # pcs = [99.99,]
    # pc = pcs[0]
    pc = 99.9
    rho = 166.1
    # rho = 28
    simulator = Lorenz63Simulator(rho=rho,rng=rng)

    # Initialize some parameters for the simulation
    truth_initial_state = [1.0, 1.0, 1.0]
    spinup_time = 500
    num_steps = daylength + spinup_time
    # num_steps = 250015  # Increase num_steps to at least 500*500 + spinup_time

    dt = 0.01
    num_perturbations = 200
    # Initial-condition perturbations (will be pos/neg)
    perturbation_scale = 5E-18

    # Max drift per time-step, uniform sampling
    # All positive to drive a consistent drift for each member from truth.
    # Might change to only do in one dimension (Z)
    drift_minmax = [1E-19,1E-18]

    # Run the simulations
    truth_series, perturbed_series_list = simulator.run_simulations(
        truth_initial_state, num_steps, dt, spinup_time, num_perturbations, perturbation_scale,
        drift_minmax=drift_minmax)

    # Compute the 99th and 99.9th percentiles
    # all_data = np.concatenate([ps[:] for ps in perturbed_series_list])
    # q99 = np.percentile(all_data, 99)
    # q999 = np.percentile(all_data, 99.9)

    # Threshold for all events
    truth_exceed = np.percentile(truth_series,pc)

    #### EXCEEDENCE AND MAX FILTERING
    #### FIGURE EXAMPLE VISUALISATIONS AND FOR TESTING
    DATA = np.zeros((num_perturbations,daylength))

    # plot_truth(truth_series,pc,ylims)
    ####### Plot truth
    truth_exc = do_exceedence(truth_series,exceed=truth_exceed,do_plot=True)

    print(f"{truth_exc.shape=}, {DATA.shape=}")

    for i, ps in enumerate(perturbed_series_list):
        # Slice the first (500*500) data points after spinup_time
        data_points = ps[:]
        pass

        if i<3:
            do_plot = True
            # Show first shape**2 in time series
            plot_time_series(data_points,ylims,exceed=truth_exceed,)

            # Plot a max filter for this, tuned to give an intermittent ts
            # max_data = plot_max_filter(data_points,ylims)

            # Plot exceedence of pc, with max filter for "danger zones" of time
        else:
            do_plot = False
        data = do_exceedence(data_points,exceed=truth_exceed ,do_plot=do_plot)

        # Create a matrix to store the exceedances
        # exceedance_matrix_99 = (data_points > q99)
        # exceedance_matrix_999 = (data_points > q999)

        # Reshape the exceedance matrices and visualize them
        # reshape_and_visualize(exceedance_matrix_99, f'Exceedance of 99th percentile for perturbation {i+1}',shape)
        # reshape_and_visualize(exceedance_matrix_999, f'Exceedance of 99.9th percentile for perturbation {i+1}',shape)

        # Now get processed data into matrix for probs
        pass
        DATA[i,:] = data

    ### DO PERCENTAGE CALCS
    probs = np.mean(DATA,axis=0)
    # n=None is no smoothing

    # smoothed in another way than before - need to move do_smooth into
    # plot_probs and rename it "plot and smooth"

    n = 200
    probs_sm = do_smooth(probs,n)
    # Try this?
    # data = moving_average(data,n)
    plot_probs(probs_sm)

    truth_sm = do_smooth(truth_exc,n)
    plot_truth(truth_sm)

    # Verify
    fig,ax = plt.subplots(1)
    ax2 = ax.twinx()

    # Check ts size
    print(f"{len(truth_sm)=}, {len(probs_sm)=}.")

    # Brier
    bs_1d = (probs_sm-truth_sm)**2
    ig_1d = scipy.special.kl_div(truth_sm,probs_sm)

    # Plot
    ax.plot(bs_1d)
    ax2.plot(ig_1d, color="red")
    ax2.set_title("Info Gain (bits)")
    fig.show()

    # sums
    bs = np.sum(bs_1d)
    ig = np.sum(ig_1d)
    print(f"{bs=}, {ig=}")

    # Do skill scores
    baserate = np.mean(truth_sm)
    UNC_bs = baserate*(1-baserate)
    bss_1d = (bs_1d-UNC_bs) / (0-UNC_bs)
    bss = np.mean(bss_1d)
    print(f"Brier skill score total is {bss}")

    # In bits!
    UNC_ig = -baserate * np.log2(baserate)
    igss_1d = (ig_1d-UNC_ig) / (0-UNC_ig)
    igss = np.mean(igss_1d)
    print(f"Info gain skill score total is {igss}")

    # Verify
    fig, ax = plt.subplots(1)
    ax2 = ax.twinx()
    ax.plot(bss_1d)
    ax2.plot(igss_1d, color="red")
    ax2.set_title("Info Gain (bits)")
    ax.set_ylim([1,-2])
    ax2.set_ylim([1,-2])
    fig.show()

    # Times exceeding thresh (True)
    ex_pc = np.mean(truth_sm) * 100
    print(f"{len(bs_1d)} units of time, with {ex_pc:.2f}pc of obs exceeding thresh.")
    pass

if __name__ == "__main__":
    main()