import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

class MultiplexTrial:
    def __init__(self) -> None:
        self.raw_data = None
        self.processed_data = None

    def load_data(self, data_path):
        self.processed_data = pd.read_csv(data_path)

    def select_test_period(self):
        df = self.processed_data
        test_df = df[df['experiment_step'] == 'Test']
        test_df = test_df.filter(regex=r'^(Timestamp|chamber_\d+_loc)$')
        test_df.set_index('Timestamp', inplace=True)
        return test_df

    def select_initial_valence_period(self):
        df = self.processed_data
        initial_valence_df = df[df['experiment_step'] == 'Initial Valence']
        initial_valence_df = initial_valence_df.filter(regex=r'^(Timestamp|chamber_\d+_loc)$')
        initial_valence_df.set_index('Timestamp', inplace=True)
        return initial_valence_df

    def filter_by_num_choices(self, midline_borders, threshold=1, filter='both'):
        valence_df = self.select_initial_valence_period()
        test_df = self.select_test_period()
        df_mapping = {
            'both': [('valence_df', valence_df), ('test_df', test_df)],
            'test': [('test_df', test_df)],
            'valence': [('valence_df', valence_df)]
        }
        filtered_dfs = {}
        for key, df in df_mapping.get(filter, []):
            filtered_dfs[f"filtered_{key}"] = self.filter_by_midline(df, midline_borders=midline_borders, threshold=threshold)
        if filter == 'both':
            common_columns = filtered_dfs['filtered_valence_df'].columns.intersection(filtered_dfs['filtered_test_df'].columns)
            self.processed_data = filtered_dfs['filtered_valence_df'][common_columns], filtered_dfs['filtered_test_df'][common_columns]
        elif filter in ['test', 'valence']:
            key = f"filtered_{filter}_df"
            self.processed_data = filtered_dfs[key][filtered_dfs[key].columns]

    def filter_by_midline(self, df, midline_borders, threshold=1):
        crossing_counts = {}
        for col in df.columns:
            values = df[col]
            crossings = (
                ((values.shift(1) < midline_borders) & (values >= midline_borders)) | 
                ((values.shift(1) > midline_borders) & (values <= midline_borders)) |
                ((values.shift(1) > -midline_borders) & (values <= -midline_borders)) |
                ((values.shift(1) < -midline_borders) & (values >= -midline_borders))
            )
            crossing_counts[col] = crossings.sum()
        filtered_columns = [col for col, count in crossing_counts.items() if count >= threshold]
        filtered_df = df[filtered_columns]
        return filtered_df

    @staticmethod
    def time_spent(df, width_size=20, sampling_rate=0.1):
        def process_counts(counts):
            df_transposed = counts.reset_index().T
            df_transposed.columns = df_transposed.iloc[0]
            return df_transposed.drop(df_transposed.index[0])
        mask_greater = df > width_size
        mask_less = df < -width_size
        count_greater = mask_greater.sum() * sampling_rate
        count_less = mask_less.sum() * sampling_rate
        count_greater_processed = process_counts(count_greater)
        count_less_processed = process_counts(count_less)
        df_combined = pd.concat([count_greater_processed, count_less_processed])
        df_combined.index = ['right_side', 'left_side']
        return df_combined

    def analyse_time(self):
        valence_df = self.time_spent(self.processed_data[0])
        test_df = self.time_spent(self.processed_data[1])
        valence_denominator = valence_df.iloc[1] + valence_df.iloc[0]
        test_denominator = test_df.iloc[0] + test_df.iloc[1]
        combined_mask = (valence_denominator != 0) & (test_denominator != 0)
        filtered_valence_df = valence_df.loc[:, combined_mask]
        filtered_test_df = test_df.loc[:, combined_mask]
        initial_val = (filtered_valence_df.iloc[0] - filtered_valence_df.iloc[1]) / (filtered_valence_df.iloc[1] + filtered_valence_df.iloc[0])
        end_valence = (filtered_test_df.iloc[1] - filtered_test_df.iloc[0]) / (filtered_test_df.iloc[0] + filtered_test_df.iloc[1])
        learned_index = (end_valence - initial_val) / 2
        return learned_index

def analyze_experiment_folder(folder_path, threshold, control_groups, experimental_groups):
    all_trials_data = pd.DataFrame()
    all_trials_data = collect_trial_data(folder_path, all_trials_data, threshold)
    all_trials_data_cleaned = clean_trial_data(all_trials_data)
    stats_results = perform_statistical_analysis(all_trials_data_cleaned, control_groups, experimental_groups)
    save_results_to_csv(folder_path, all_trials_data_cleaned, stats_results)
    create_and_save_plots(folder_path, all_trials_data_cleaned, stats_results)

def collect_trial_data(folder_path, all_trials_data, threshold):
    for date_folder in os.listdir(folder_path):
        date_path = os.path.join(folder_path, date_folder)
        if os.path.isdir(date_path):
            for trial_folder in os.listdir(date_path):
                trial_path = os.path.join(date_path, trial_folder)
                if os.path.isdir(trial_path):
                    metadata_path = os.path.join(trial_path, 'experiment_metadata.json')
                    data_path = os.path.join(trial_path, 'fly_loc.csv')
                    if os.path.exists(metadata_path) and os.path.exists(data_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        fly_genotype = metadata.get('flyGenotype')
                        trial = MultiplexTrial()
                        trial.load_data(data_path)
                        trial.filter_by_num_choices(midline_borders=0.6, threshold=threshold, filter='both')
                        learned_index = trial.analyse_time()
                        trial_data = pd.DataFrame({fly_genotype: learned_index})
                        all_trials_data = pd.concat([all_trials_data, trial_data], ignore_index=True)
    return all_trials_data

def clean_trial_data(all_trials_data):
    return all_trials_data.apply(lambda x: x.dropna().reset_index(drop=True))

def perform_statistical_analysis(all_trials_data, control_groups, experimental_groups, num_permutations=10000):
    """
    Perform statistical analysis only between each experimental group and its relevant controls:
    - Each experimental group is compared to its corresponding control (specific and baseline).
    """
    stats_results = []

    for exp in experimental_groups:
        # Extract the genotype root from the experimental group name
        exp_genotype = exp.split('x')[0]
        exp_specific_part = exp.split('x')[1]  # e.g., 'mb247' in '33885xmb247'
        
        # Determine relevant controls:
        relevant_controls = [
            ctrl for ctrl in control_groups 
            if (exp_genotype in ctrl) or (exp_specific_part in ctrl)
        ]

        for control in relevant_controls:
            # Extract data for experimental and control groups
            control_data = pd.to_numeric(all_trials_data[control], errors='coerce').dropna()
            experimental_data = pd.to_numeric(all_trials_data[exp], errors='coerce').dropna()

            # Perform the permutation test
            observed_diff, p_value = permutation_test(control_data, experimental_data, num_permutations=num_permutations)
            
            # Apply Tukey-like correction for multiple comparisons
            corrected_p_value = min(p_value * len(relevant_controls), 1.0)

            # Append results for this comparison
            stats_results.append({
                'Comparison': f'{exp} vs {control}',
                'Test Type': "Permutation Test",
                'Test Statistic': observed_diff,
                'Original p-value': p_value,
                'Corrected p-value': corrected_p_value
            })

    stats_results_df = pd.DataFrame(stats_results)
    return stats_results_df



def permutation_test(group1, group2, num_permutations=1000):
    combined = np.concatenate([group1, group2])
    observed_diff = np.mean(group1) - np.mean(group2)
    permuted_diffs = np.zeros(num_permutations)
    for i in range(num_permutations):
        np.random.shuffle(combined)
        perm_group1 = combined[:len(group1)]
        perm_group2 = combined[len(group1):]
        permuted_diffs[i] = np.mean(perm_group1) - np.mean(perm_group2)
    p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
    return observed_diff, p_value

def save_results_to_csv(folder_path, all_trials_data_cleaned, stats_results):
    output_folder = os.path.join(folder_path, 'output')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_csv_path = os.path.join(output_folder, 'experiment_results_cleaned.csv')
    all_trials_data_cleaned.to_csv(output_csv_path, index=False)
    output_stats_path = os.path.join(output_folder, 'statistical_results.csv')
    stats_results.to_csv(output_stats_path, index=False)

def create_and_save_plots(folder_path, all_trials_data_cleaned, stats_results):
    all_trials_data_long = all_trials_data_cleaned.melt(var_name='Genotype', value_name='Learned Index')
    output_folder = os.path.join(folder_path, 'output', 'plots')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    combined_fig_path = os.path.join(output_folder, 'combined_experiment_plots.png')
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    create_bar_plot(all_trials_data_long, stats_results)
    plt.subplot(1, 3, 2)
    create_box_plot(all_trials_data_long)
    plt.subplot(1, 3, 3)
    create_swirl_plot(all_trials_data_long)
    plt.tight_layout()
    plt.savefig(combined_fig_path, dpi=300)
    plt.clf()
    save_individual_plots(output_folder, all_trials_data_long, stats_results)

def save_individual_plots(output_folder, data, stats_results):
    bar_plot_path = os.path.join(output_folder, 'bar_plot.png')
    plt.figure(figsize=(6, 5))
    create_bar_plot(data, stats_results)
    plt.savefig(bar_plot_path, dpi=300)
    plt.clf()
    box_plot_path = os.path.join(output_folder, 'box_plot.png')
    plt.figure(figsize=(6, 5))
    create_box_plot(data)
    plt.savefig(box_plot_path, dpi=300)
    plt.clf()
    swirl_plot_path = os.path.join(output_folder, 'swirl_plot.png')
    plt.figure(figsize=(6, 5))
    create_swirl_plot(data)
    plt.savefig(swirl_plot_path, dpi=300)
    plt.clf()

def create_bar_plot(data, stats_results):
    # Calculate sample sizes
    sample_sizes = calculate_sample_size(data)

    # Plot the error bars first with a lower zorder
    sns.barplot(data=data, x='Genotype', y='Learned Index', hue='Genotype', palette="deep", errorbar="se", estimator="mean", capsize=0.1, zorder=5)
    
    # Plot the mean bars again without error bars with a higher zorder
    sns.barplot(data=data, x='Genotype', y='Learned Index', hue='Genotype', palette="deep", errorbar=None, estimator="mean", edgecolor=".2", zorder=10)

    # Add statistical significance annotations
    add_statistical_annotations(stats_results)

    # Add a custom legend with sample sizes
    sample_size_labels = [f"{genotype}: n={count}" for genotype, count in sample_sizes.items()]
    plt.legend(title="Sample Sizes", labels=sample_size_labels, loc='upper right', bbox_to_anchor=(1, 1))

    plt.ylim(-1, 1)
    plt.title('Bar Plot of Learned Index')
    plt.ylabel('Learned Index')
    plt.xlabel('Genotype')



def create_box_plot(data):
    # Calculate sample sizes
    sample_sizes = calculate_sample_size(data)

    sns.boxplot(data=data, x='Genotype', y='Learned Index', palette="deep", hue='Genotype')
    
    # Add a custom legend with sample sizes
    sample_size_labels = [f"{genotype}: n={count}" for genotype, count in sample_sizes.items()]
    plt.legend(title="Sample Sizes", labels=sample_size_labels, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.ylim(-1, 1)
    plt.title('Box Plot of Learned Index')
    plt.ylabel('Learned Index')
    plt.xlabel('Genotype')



def create_swirl_plot(data):
    # Calculate sample sizes
    sample_sizes = calculate_sample_size(data)

    sns.swarmplot(data=data, x='Genotype', y='Learned Index', palette="deep", hue="Genotype")
    
    # Add a custom legend with sample sizes
    sample_size_labels = [f"{genotype}: n={count}" for genotype, count in sample_sizes.items()]
    plt.legend(title="Sample Sizes", labels=sample_size_labels, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.ylim(-1, 1)
    plt.title('Swirl Plot of Learned Index')
    plt.ylabel('Learned Index')
    plt.xlabel('Genotype')


    
def add_statistical_annotations(stats_results):
    significance_threshold = 0.05
    for index, row in stats_results.iterrows():
        p_value = row['Corrected p-value']  # Use 'Corrected p-value' instead of 'Test p-value'
        if p_value < significance_threshold:
            plt.text(index, 0.9, '*', ha='center', va='bottom', color='black', fontsize=20)

def calculate_sample_size(data):
    """
    Calculate the sample size for each genotype.
    Returns a dictionary with genotype names as keys and sample sizes as values.
    """
    sample_sizes = data.dropna().groupby('Genotype').size()
    return sample_sizes.to_dict()


# Example Usage:
analyze_experiment_folder(
    folder_path='G:/My Drive/Work/PhD Neuroscience/Moshe Parnas/Experiments/Serotonergic system/5ht_behavior/raw_data/multiplex/5ht_receptors_knockdown_classical/5ht1a_rnai', 
    threshold=4, 
    control_groups=['w1118x33885', 'w1118xmb247'], 
    experimental_groups=['33885xmb247', '33885xmb504']
)
