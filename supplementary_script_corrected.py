import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# This notebook will define information gain and brier score
# We will generate a synthetic time series of a rare event - forecast-observation pairs for a sequence of time samples
# Two forecasting systems give probabilistic occurrence of this rare event
# There's an old (worse) and new (better) model so we can compute the gain (or loss) in information and Brier Score 
# Both models are overconfident and under-predict the rare event

# We ignore observational error with the observations 0 or 1
# Forecasts are between 0 and 1 (probability) but never 0 or 1 due to it breaking logarithms

# There are 100,000 forecast-observation pairs from these fake models for the same event time series
# This is a column called "o"
# The old model's forecast is in a column called "f_old"
# The new model's forecast is in a column called "f_new"
# The model forecasts are roughly correlated with the observations so we have a realistic reliability diagram
# There is more variation in the f_old forecasts than the f_new forecasts so the reliability diagram is more curved for the old model
# The old model is more overconfident than the new model so the old model has a higher Brier Score and lower information gain

# Assuming the observation array 'o' is defined
# Predefined set of forecast probabilities
p_k = np.array([0.005,0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99,0.995])
rng = np.random.default_rng(27)
N = int(1E5)  # Number of forecast-observation pairs
base_rate = 0.005
# Setting the error too low might actually cause lower skill - can check sensitivity TODO!
error_old_f = 0.1
error_new_f = 0.03
autocorr = 0.8
# window length for plotting time series
nn = 35

class DKL:
    """
    This class is used to compute the Ignorance and Brier Score for two forecasting systems.
    Another function will compute information gain using one model or prior entropy (uncertainty component) as baseline
    It takes as input the forecasts and observations to compute various metrics.
    Brier Scores are included as components of the information-based methods
    """
    def __init__(self, f, o, binning_k):

        """
        Initialize the class with forecasts and observations pairs.

        Parameters:
        f (array-like): Forecasts
        o (array-like): Observations corresponding to the forecasts.
        binning_k (array-like): Probability bins for quantising the forecasts.        
        """
        self.f = f
        self.o = o

        # Probability bins
        if binning_k is None:
            # This might not work without interpolation to these default values
            # Also creates bins for 0 and 1!
            self.p_k = np.linspace(0, 1, num=11)
        else:
            self.p_k = binning_k
        
        # The number of total forecast-observation pairs
        self.N = len(self.o)

        # Base rate, o_bar, of event
        self.o_bar = np.nanmean(self.o)

        # Generate binning of forecasts and observations
        self.df = self.do_binning()

    def compute_ign(self):
        # Ignorance is just DKL - this is the method 
        ign = np.mean(self.compute_dkl(self.o, self.f))
        return ign

    @staticmethod
    def compute_dkl(q, p):
        """
        Compute the Kullback-Leibler divergence, handling edge cases explicitly.
        
        Parameters:
        q (float or array-like): True probabilities.
        p (float or array-like): Forecast probabilities.
        
        Returns:
        float or array-like: The Kullback-Leibler divergence, ensuring floating-point output.
        """
        # Ensure q and p are numpy arrays for element-wise operations
        q = np.asarray(q, dtype=float)
        p = np.asarray(p, dtype=float)
        
        # Initialize DKL with zeros in the same shape as q and p, explicitly using a floating-point type
        dkl = np.zeros_like(q, dtype=float)
        
        # Compute DKL for valid indices (where q != 0 and q != 1)
        valid_idx = (q != 0) & (q != 1)
        dkl[valid_idx] = q[valid_idx] * np.log2(q[valid_idx] / p[valid_idx]) + \
                         (1 - q[valid_idx]) * np.log2((1 - q[valid_idx]) / (1 - p[valid_idx]))
        
        # Handle edge cases: q == 0 or q == 1 specifically
        q_zero_idx = q == 0
        dkl[q_zero_idx] = (1 - q[q_zero_idx]) * np.log2((1 - q[q_zero_idx]) / (1 - p[q_zero_idx]))
        
        q_one_idx = q == 1
        dkl[q_one_idx] = q[q_one_idx] * np.log2(q[q_one_idx] / p[q_one_idx])
        
        # Set divergences to infinity where appropriate, now safe with dtype=float
        dkl[(p == 0) & (q > 0)] = np.inf
        dkl[(p == 1) & (q < 1)] = np.inf
        
        return dkl

    def compute_unc_ign(self):
        # Compute the entropy of the observations
        if self.o_bar == 0 or self.o_bar == 1:
            unc = 0
        else:
            unc = -(self.o_bar * np.log2(self.o_bar) + (1-self.o_bar) * np.log2(1-self.o_bar))
        return unc
    
    def compute_ignss(self):
        # IGNSS is the ignorance skill score
        # It is positive if the forecast is better than climatology and negative if it is worse
        ign = self.compute_ign()
        unc = self.compute_unc_ign()
        if unc == 0:
            return np.nan
        ignss = self.compute_skill_score(ign, unc)
        return ignss

    def do_binning(self):
        # Define bins for forecast probabilities
        bin_labels = self.p_k
        # The bin edges should be the midpoints between the bin labels
        bin_edges = (bin_labels[:-1] + bin_labels[1:]) / 2
        # bin_labels = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins as labels

        # Categorize each forecast into a bin
        f_bin_indices = np.digitize(self.f, bin_edges, right=False) 
        # Initialize counters for each bin
        n_k = np.zeros_like(bin_labels)
        o_k = np.zeros_like(bin_labels)

        # Populate n_k and o_k
        for i, label in enumerate(bin_labels):
            # Indices where forecasts fall into the current bin
            in_bin = f_bin_indices == i
            n_k[i] = np.sum(in_bin)
            o_k[i] = np.sum(self.o[in_bin])

        # Construct and return the DataFrame
        bins_df = pd.DataFrame({
            'k': np.arange(len(bin_labels)),
            'p_k': bin_labels,
            'o_k': o_k,
            'n_k': n_k
        })

        return bins_df

    def compute_rel_ign(self):
        total_dkl = 0
        # Iterate over each row in the DataFrame
        for _, row in self.df.iterrows():
            p_k = row['p_k']  # Probability forecast for the bin
            o_k = row['o_k']  # Number of observations of the event in this bin
            n_k = row['n_k']  # Number of forecasts in this bin
            
            # Ensure non-zero n_k to avoid division by zero
            if n_k > 0:
                ok_bar = o_k / n_k  # Observed frequency of the event in this bin
                # Compute DKL using the observed frequency and the forecast probability
                dkl = self.compute_dkl(ok_bar, p_k)
                total_dkl += dkl * n_k
    
        # Return the total DKL divided by the total number of observations
        return total_dkl / self.N

    def compute_res_ign(self):
        total_dkl = 0
        # Overall observed frequency of the event
        o_bar = np.sum(self.df['o_k']) / self.N
    
        # Iterate over each row in the DataFrame
        for _, row in self.df.iterrows():
            p_k = row['p_k']  # Probability forecast for the bin
            o_k = row['o_k']  # Number of observations of the event in this bin
            n_k = row['n_k']  # Number of forecasts in this bin
    
            # Ensure non-zero n_k to avoid division by zero
            if n_k > 0:
                ok_bar = o_k / n_k  # Observed frequency of the event in this bin
                # Compute DKL using the observed frequency and the overall observed frequency
                dkl = self.compute_dkl(ok_bar, o_bar)
                total_dkl += dkl * n_k
    
        # Return the total DKL divided by the total number of observations
        return total_dkl / self.N

    def compute_unc_bs(self):
        return self.o_bar * (1 - self.o_bar)

    def compute_bs(self):
        # Compute brier score for self.o and self.f
        bs = np.nanmean(self.compute_bs_ts(self.o,self.f))
        return bs
    
    @staticmethod
    def compute_bs_ts(o,f):
        # Compute brier score for each time step
        bs = (o - f)**2
        return bs

    def compute_bss(self):
        # Compute brier score skill score
        bs = self.compute_bs()
        unc_bs = self.compute_unc_bs()
        bss = self.compute_skill_score(bs, unc_bs)
        # bss = (bs - unc_bs) / (0-unc_bs)
        return bss
    
    @staticmethod
    def compute_skill_score(f, b):
        # Compute the skill score of with "f" being forecast and "b" baseline
        # Positive means 'f' is better than 'b'
        # Negative means forecast model has less skill than the baseline
        ss = (f-b)/(0-b)
        return ss
    
    def ign_from_components(self, prints=False):
        rel = self.compute_rel_ign()
        res = self.compute_res_ign()
        unc = self.compute_unc_ign()
        ign = unc + rel - res
        ignss = self.compute_skill_score(ign, unc)
        if prints:
            df = pd.DataFrame({
                'Reliability': [rel],
                'Resolution': [res],
                'Uncertainty': [unc],
                'Ignorance': [ign],
                'Ignorance Skill Score': [ignss]
            })
            return df 
        return ign
    
    def compute_components(self,prints=False):
        rel = self.compute_rel_ign()
        res = self.compute_res_ign()
        unc = self.compute_unc_ign()
        ign = self.compute_ign()
        brier = self.compute_bs()
        ignss = self.compute_ignss()
        bss = self.compute_bss()
        
        if prints:
        # Create a DataFrame for pretty printing
            df = pd.DataFrame({
                'Reliability': [rel],
                'Resolution': [res],
                'Uncertainty': [unc],
                'Ignorance': [ign],
                'Ignorance Skill Score': [ignss],
                'Brier Score': [brier],
                'Brier Skill Score': [bss]
            })
            # print(df)
            return df
        
        return rel, res, unc, ign, ignss, brier, bss
    
def calculate_info_gain(ign_baseline, ign_new):
    """
    Calculate the information gain between a baseline and a new forecast model.
    
    Parameters:
    - ign_baseline (float): Ignorance of the baseline forecast model.
    - ign_new (float): Ignorance of the new forecast model.
    
    Returns:
    - float: Information gain of the new forecast model over the baseline.
    """
    return ign_baseline - ign_new



def generate_time_series(rng, n=10000, storm_prob=0.01, forecast_error=0.01, autocorr=0.9, o=None):
    """
    Generates a time series for thunderstorm occurrence and probabilistic forecasts with improved correlation,
    using numpy's random number Generator for consistent reproducibility. Allows for an optional observed
    series 'o' to be provided.
    
    Parameters:
    - n (int): Length of the time series.
    - storm_prob (float): Probability of a thunderstorm occurring.
    - forecast_error (float): Standard deviation of the forecast error.
    - autocorr (float): Autocorrelation factor for both observed and forecasted series.
    - seed (int): Seed for the random number generator for reproducibility.
    - o (pandas.Series or None): Optional observed occurrence series. If None, 'o' is generated as per the original method.
    
    Returns:
    - DataFrame with columns 'f' for forecast probability and 'o' for observed occurrence (0 or 1).
    """   
    # if o is a pandas series or dataframe, make it numpy array 
    if isinstance(o, pd.Series) or isinstance(o, pd.DataFrame):
        o = o.to_numpy()
    
    # Check if 'o' is provided, if not, generate 'o' as before
    if o is None:
        o = rng.random(n) < storm_prob
        o = np.round(o).astype(int)  # Ensure binary outcomes

        # Introduce autocorrelation in observations
        for i in range(1, n):
            if rng.random() < autocorr:
                o[i] = o[i-1]
    else:
        # Ensure 'o' is a binary series as expected
        o = o.astype(int)
        n = len(o)  # Update 'n' based on the length of provided 'o'

    f = np.zeros(n)  # Initialize forecast probabilities array

    # Adjust initial forecast based on observations
    for i in range(1, n):
        base_forecast = storm_prob if o[i-1] == 0 else 1 - storm_prob if isinstance(o, pd.Series) else storm_prob if o[i-1] == 0 else 1 - storm_prob
        adjustment = rng.normal() * forecast_error
        
        # Incorporate autocorrelation in forecasts
        if rng.random() < autocorr and i > 1:
            f[i] = f[i-1] + adjustment
        else:
            f[i] = base_forecast + adjustment

        # Ensure forecasts are within [0, 1] range
        f[i] = np.clip(f[i], 0, 1)

    df = pd.DataFrame({'f': f, 'o': o})
    
    return df


def quantise_forecasts(df, bins):
    """
    Quantises forecast probabilities into specified bins. Ensures 0% goes into the "0.005" bin and 100%
    goes into the "0.995" bin, with the rest following the specified bins.
    
    Parameters:
    - df (DataFrame): DataFrame containing 'f' column for forecast probabilities.
    - bins (tuple): Tuple of probability bins.
    
    Returns:
    - DataFrame with quantised forecast probabilities in 'f_quantised' column.
    """
    print("Max and min of forecast probabilities in df dataframe for 'f' column")
    print(df['f'].max(), df['f'].min())
       
    
    # Correct handling for 0% and 100% forecasts to align with the nearest bins
    df['f'] = np.where(df['f'] <= 0, 0.005, df['f'])  # Adjust to 0.005 for 0%
    df['f'] = np.where(df['f'] >= 1, 0.995, df['f'])  # Adjust to 0.995 for 100%

    bins = np.array(bins)  # Ensure bins is a numpy array for digitize function

    # Find the index of the closest bin for each forecast, after handling 0% and 100% cases
    idx = np.digitize(df['f'], bins, right=True)  # Consider if right=True aligns with desired binning behavior

    # Replace forecast probabilities with bin labels, ensuring adjustments for 0 and 1 are considered
    df['f_quantised'] = bins[np.clip(idx - 1, 0, len(bins) - 1)]  # Adjust index and ensure it's within bounds
    
    return df


def plot_df_distr(df, p_k):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the observed frequency and number of forecasts for each bin
    ax.bar(p_k, df['o_k'], width=0.01, align='center', label='Observed', color='red', alpha=0.7)
    ax.bar(p_k, df['n_k'], width=0.01, align='center', label='Forecast', color='blue', alpha=0.7)

    # Set the title and labels
    ax.set_title('Observed Frequency and Number of Forecasts for Each Bin')
    ax.set_xlabel('Forecast Probability')
    ax.set_ylabel('Frequency')
    ax.legend()

    # Show thin vertical lines at each unique value of p_k (forecast bins)
    for p in p_k:
        ax.axvline(p, color='black', alpha=0.2)

    # Display the plot
    plt.show()


# Write a function that generate the time series then quantise the forecasts according to p_k
def generate_and_quantise_time_series(n, storm_prob, forecast_error, autocorr, p_k, o=None):
    """
    Generates a time series of forecast probabilities and observed occurrences, and quantises the forecasts
    into specified probability bins.
    
    Parameters:
    - n (int): Length of the time series.
    - storm_prob (float): Probability of a thunderstorm occurring.
    - forecast_error (float): Standard deviation of the forecast error.
    - autocorr (float): Autocorrelation factor for both observed and forecasted series.
    - p_k (array-like): Probability bins for quantising the forecasts.
    
    Returns:
    - DataFrame with columns 'f' for forecast probability, 'o' for observed occurrence, and 'f_quantised' for quantised forecasts.
    """
    # Generate the time series
    df_raw = generate_time_series(rng, n, storm_prob, forecast_error, autocorr, o=o)
    
    # Quantise the forecasts
    df_quantised = quantise_forecasts(df_raw, p_k)
    
    fig, ax = plt.subplots(1, figsize=(7, 6))  
    
    # Plot the quantised forecast probabilities on the second subplot
    sns.histplot(df_quantised['f_quantised'], bins=p_k, kde=False, ax=ax)
    ax.set_title('Quantised Forecast Probabilities')
    
    # label axes
    ax.set_xlabel("Probability")
    ax.set_ylabel("Count")
    
    plt.tight_layout()  # Adjust the padding between and around the subplots
    plt.show()
    
    return df_raw, df_quantised

def plot_time_series(df, col, window_size):
    """
    Plots a time series of forecast probabilities and observed occurrences.
    
    Parameters:
    - df (DataFrame): DataFrame containing 'f' and 'o' columns.
    - window_size (int): Number of rows to plot around the first occurrence.
    """
    # Create a mini_df that subsets the df to a window around the 10th occurrence where df['o'] is 1
    # mini_df = ?
    event_indices = df[df['o'] == 1].index
    event_index = event_indices[9]  
    mini_df = df.iloc[event_index - window_size:event_index + window_size]

    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the forecast probabilities
    ax.plot(mini_df.index, mini_df[col], label='Forecast', color='blue')
    
    # Plot the observed occurrences
    ax.scatter(mini_df.index, mini_df['o'], label='Observation', color='red')
    
    # Set the title and labels
    ax.set_title('Time Series of Forecast Probabilities and Observations')
    ax.set_xlabel('Time')
    ax.set_ylabel('Probability / Occurrence')
    ax.legend()
    
    ax.set_ylim(0, 1)
    
    # Show thin horizontal lines at each unique value of f_quantised (forecast bins)
    for p in p_k:
        ax.axhline(p, color='black', alpha=0.2)
        
    # Fill the graph area between y=0 and y=1 when the observation is 1 in red with a high alpha
    ax.fill_between(mini_df.index, 0, 1, where=mini_df['o'] == 1, color='red', alpha=0.1)
    
    # Display the plot
    plt.show()
    return
    

def plot_time_series_around_first_event(df, col, window_size, score="dkl", ax=None):
    """
    Plots a time series of forecast probabilities and observed occurrences.

    Parameters:
    - df (DataFrame): DataFrame containing 'f' and 'o' columns.
    - col (str): Column name to plot.
    - window_size (int): Number of rows to plot around the first occurrence.
    - score (str): Score to compute for the time series. Can be "dkl" or "brier".
    - ax (matplotlib.axes.Axes): The axes to draw on. If None, create a new figure and axes.
    """
    if score == "dkl":
        verif_ts = DKL.compute_dkl(df['o'],df[col])
    elif score == "brier":
        verif_ts = DKL.compute_bs_ts(df['o'],df[col])
    else:
        print("Score not recognised")
        raise Exception

    # Find the index of the first occurrence where df['o'] is 1
    event_index = df[df['o'] == 1].index[0]

    # Create a mini_df that subsets the df to a window around the first occurrence
    mini_df = df.iloc[event_index - window_size:event_index + window_size]

    # If no axes object is provided, create a new figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the forecast probabilities
    ax.plot(mini_df.index, mini_df[col], label='Forecast', color='blue')

    # Plot the observed occurrences
    ax.scatter(mini_df.index, mini_df['o'], label='Observation', color='red')

    # Set the title and labels
    ax.set_title('Time Series of Forecast Probabilities and Observations')
    ax.set_xlabel('Time')
    ax.set_ylabel('Probability / Occurrence')

    ax.set_ylim(0, 1)

    # On a second axis, plot dkl_ts for the same time
    ax2 = ax.twinx()
    ax2.plot(mini_df.index, verif_ts[event_index - window_size:event_index + window_size], label=score, color='green', linestyle='--')
    # axis label is score + units (DKL is bits and Brier Score is probability - units are not shown)
    score_str = score + " (bits)" if score == "dkl" else score
    ax2.set_ylabel(score_str, color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Set legend for all three lines
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Show thin horizontal lines at each unique value of f_quantised (forecast bins)
    for p in p_k:
        ax.axhline(p, color='black', alpha=0.2)

    # Fill the graph area between y=0 and y=1 when the observation is 1 in red with a high alpha
    ax.fill_between(mini_df.index, 0, 1, where=mini_df['o'] == 1, color='red', alpha=0.1)

    # Return the axes object
    return ax

def plot_time_series_multi_scores(df, col, window_size, scores):
    # Create a figure with two subplots (two rows, one column)
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # For each score in scores
    for i, score in enumerate(scores):
        # Plot the time series around the first event on the subplot
        plot_time_series_around_first_event(df, col, window_size, score=score, ax=axs[i])

        # Label the subplot
        axs[i].set_title(f'({chr(97 + i)}) {score.capitalize()}')

    # Adjust the padding between and around the subplots
    plt.tight_layout()

    # Display the plot
    plt.show()



def plot_reliability(df, title='Reliability Diagram', ax=None):
    """
    Plots the reliability diagram for a given dataframe.

    Parameters:
    - df (DataFrame): DataFrame containing 'p_k', 'o_k', and 'n_k' columns.
    - title (str): Title of the plot.
    - ax (matplotlib.axes.Axes): The axes to draw on. If None, create a new figure and axes.

    Returns:
    - matplotlib.axes.Axes: The axes with the plot.
    """
    # Calculate observed frequency for each bin
    df['observed_frequency'] = df['o_k'] / df['n_k']

    # If no axes object is provided, create a new figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting the reliability diagram
    ax.plot(df['p_k'], df['observed_frequency'], marker='o', linestyle='-', color='blue', label='Observed Frequency')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Reliability')

    # Formatting the plot
    ax.set_title(title)
    ax.set_xlabel('Forecast Probability (p_k)')
    ax.set_ylabel('Observed Frequency')
    ax.legend(loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    return ax


def plot_histograms_on_same_axes(dkl_old, dkl_new, threshold, title, plot_poss_values=True):
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Subset of DKL values for better visualization
    dkl_old_sub = dkl_old[dkl_old > threshold]
    dkl_new_sub = dkl_new[dkl_new > threshold]

    # Calculate the common bins for both histograms
    combined_data = np.concatenate([dkl_old_sub, dkl_new_sub])
    bins = np.histogram_bin_edges(combined_data, bins='auto')

    # Calculate histogram counts for old and new DKL values
    counts_old, _ = np.histogram(dkl_old_sub, bins=bins)
    counts_new, _ = np.histogram(dkl_new_sub, bins=bins)

    # Calculate the width of each bin
    bin_widths = np.diff(bins)

    # Plot the histograms side-by-side with adjusted bar widths
    bar_width_factor = 0.95  # Adjust this factor to change the bar width
    offset = bin_widths * bar_width_factor  # Calculate the offset for the second set of bars

    ax.bar(bins[:-1], counts_old, width=bin_widths * bar_width_factor, align='edge', alpha=0.7, color='blue',
           edgecolor='black', label='Old Model', zorder=2)
    ax.bar(bins[:-1] + offset, counts_new, width=bin_widths * bar_width_factor, align='edge', alpha=0.7, color='red',
           edgecolor='black', label='New Model', zorder=2)

    # Set the title and labels
    ax.set_title(f'{title} (Threshold: {threshold})')
    ax.set_xlabel('DKL (bits)')
    ax.set_ylabel('Frequency')
    ax.legend()

    if plot_poss_values:
        vlines = [y for y in -np.log2(p_k) if y > threshold]  # Assuming p_k is defined
        for v in vlines:
            ax.axvline(x=v, color='green', linestyle='--', label='Maximum DKL' if v == vlines[0] else "", zorder=1)

    # Increase zorder to ensure bars are on top
    ax.bar(bins[:-1], counts_old, width=bin_widths * bar_width_factor, align='edge', alpha=0.7, color='blue',
           edgecolor='black', label='Old Model', zorder=3)
    ax.bar(bins[:-1] + offset, counts_new, width=bin_widths * bar_width_factor, align='edge', alpha=0.7, color='red',
           edgecolor='black', label='New Model', zorder=3)

    # Set a logarithmic scale on the y-axis
    ax.set_yscale('log')

    # Display the plot
    plt.show()

# Generate old and new quantised time series 
df_old_raw, df_old = generate_and_quantise_time_series(N, base_rate, error_old_f, autocorr, p_k)
df_new_raw, df_new = generate_and_quantise_time_series(N, base_rate, error_new_f, autocorr, p_k, o=df_old['o'])

# Check that observations (column o) in both df_ts_old_q and df_ts_new_q are the same
# If there is an exception, then print the first 10 rows of each dataframe   
if not df_old['o'].equals(df_new['o']):
    print("Observations are different in the two dataframes")
else:
    print("Observations are the same in both dataframes")

# Create a dataframe with p_k index and the number of forecasts that fall into each bin for the df_old_raw dataframe
_df = df_old_raw['f'].value_counts(bins=p_k, sort=False)
_df

plot_time_series(df_old_raw,"f", nn)

plot_time_series(df_old,"f_quantised",nn)

plot_time_series(df_new,"f_quantised",nn)


# Use the staticmethod compute_dkl to compute the DKL for the same forecast and observation period shown above
# This is the DKL for the old model
# plot_time_series_around_first_event(df_old, 'f_quantised', nn, score='dkl')


# Create a 2-panel plot for df_old with score set to 'brier' and 'dkl'
plot_time_series_multi_scores(df_old, 'f_quantised', nn, ['brier', 'dkl'])

# Create a 2-panel plot for df_new with score 'brier' and 'dkl'
plot_time_series_multi_scores(df_new, 'f_quantised', nn, ['brier', 'dkl'])

# plot_time_series_around_first_event(df_old, 'f_quantised', nn, score='brier')


# This is the DKL for the new model
# plot_time_series_around_first_event(df_new, 'f_quantised', nn, score='dkl')

# Now plot against Brier Score
# plot_time_series_around_first_event(df_new, 'f_quantised', nn, score='brier')

# Check pre-quantisised forecast probabilities for sanity
ig_old_raw = DKL(df_old['f'], df_old['o'], binning_k=p_k)
ig_old_raw.df 

# Now compute information gain of the new system over the old 
ig_old = DKL(df_old['f_quantised'], df_old['o'], binning_k=p_k)
ig_new = DKL(df_new['f_quantised'], df_new['o'], binning_k=p_k)


ig_old.df

ig_new.df

ig_old.ign_from_components(prints=True)


ig_new.ign_from_components(prints=True)

df_old_stats = ig_old.compute_components(prints=True)
df_old_stats

df_new_stats = ig_new.compute_components(prints=True)
df_new_stats

### These are the statistics 
# Combine the two dataframes into one for comparison
df_stats = pd.concat([df_old_stats, df_new_stats], axis=0)
# set the index as a string for the model names, "old" and "new"
df_stats.index = ['Old', 'New']
df_stats

### These are the forecast time series
f_old = df_old['f_quantised']
f_new = df_new['f_quantised']

#### These are the binned statistics from the dkl object for the old and new models 
df_old_binned = ig_old.df
df_new_binned = ig_new.df
df_old_binned

# Bar chart of reliability, resolution, ignorance, and ignorance skill score for new versus old models
fig, ax = plt.subplots(figsize=(13, 7))  # Adjust the figure size
df_stats.plot(kind='bar', ax=ax, width=0.8)  # Adjust the bar width
plt.title('Model Statistics Comparison')
plt.ylabel('Score')
# Set y-axis limits
miny, maxy = (-0.5,0.5)
plt.ylim(miny, maxy)

# Annotate the bars with their respective values
for p in ax.patches:
    height = p.get_height()
    if height > maxy:
        height = maxy/1.2
    elif height < miny:
        height = miny/1.2
    ax.annotate(format(p.get_height(), '.2f'),
                (p.get_x() + p.get_width() / 2., height),
                ha = 'center',
                va = 'center',
                xytext = (0, 10),
                textcoords = 'offset points')

# Put the legend top right in three columns
plt.legend(loc='upper right', ncol=3)

# Label the x-axis with the model names
ax.set_xticklabels(['Old', 'New'], rotation=0)

plt.show()

# plot_reliability(df_old_binned, title='Reliability Diagram for Old Model')
# plot_reliability(df_new_binned, title='Reliability Diagram for New Model')

# Create a figure with two subplots (two rows, one column)
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Plot the reliability diagrams on the subplots
plot_reliability(df_old_binned, title='Reliability Diagram for Old Model', ax=axs[0])
plot_reliability(df_new_binned, title='Reliability Diagram for New Model', ax=axs[1])

# Adjust the padding between and around the subplots
plt.tight_layout()

# Save the figure as a PNG file
# plt.savefig('my_figure.png')
plt.show()

# Compute information gain of the new system over the old but using the observation and forecast time series
# Do it 'event-wise' so each value is the gain over the old 
# First compute DKL over the whole time series
dkl_old = DKL.compute_dkl(df_old['o'],df_old['f_quantised'])
dkl_new = DKL.compute_dkl(df_new['o'],df_new['f_quantised'])
ig = calculate_info_gain(dkl_old, dkl_new)
ig

def plot_dkl_histogram(dkl_old, threshold, p_k, plot_poss_values=False):
    # Subset of DKL values for better visualization
    dkl_sub = dkl_old[dkl_old > threshold]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the histogram
    ax.hist(dkl_sub, bins='auto', alpha=0.7, color='blue', edgecolor='black')

    # Add the maximum possible DKL value with a vertical red line
    if plot_poss_values:
        vlines = [y for y in -np.log2(p_k) if y > threshold]
        for v in vlines:
            ax.axvline(x=v, color='red', linestyle='--', label='Maximum DKL')

    # Set the title and labels
    ax.set_title('DKL Histogram')
    ax.set_xlabel('DKL (bits)')
    ax.set_ylabel('Frequency')

    # Display the plot
    plt.show()
    



# Plot the histograms for the old and new models with threshold = 0.7
plot_histograms_on_same_axes(dkl_old, dkl_new, 0.7, 'DKL Histogram')


# Show histogram of information gain 
# Choose bins automatically

fig, ax = plt.subplots(figsize=(10, 6))
# ig_sub = ig[ig > 0.1]  # Subset of information gain values for better visualization
ig_sub = ig 
n, bins, patches = ax.hist(ig_sub, bins=50, alpha=0.7, color='blue', edgecolor='black')
# Add KDE 
# sns.kdeplot(ig, color='blue', ax=ax)
ax.set_title('Information Gain Histogram')
ax.set_xlabel('Information Gain')
ax.set_ylabel('Frequency')
plt.show()