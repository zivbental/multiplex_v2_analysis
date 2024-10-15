import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import kruskal
import statsmodels.api as sm
from statsmodels.formula.api import ols

class MultiplexTrial:
    def __init__(self) -> None:
        self.raw_data = None
        self.processed_data = None

    def load_data(self, data_path):
        """
        This function loads a Multiplex log file (.csv) and converts it to pandas dataframe.
        """
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
        """
        This function analyzes the learned behavior index for a given trial.
        It returns the learned index and its mean, which can be used in further analysis.
        """
        # Calculate time spent during the valence and test phases
        valence_df = self.time_spent(self.processed_data[0])
        test_df = self.time_spent(self.processed_data[1])

        # Calculate denominators
        valence_denominator = valence_df.iloc[1] + valence_df.iloc[0]
        test_denominator = test_df.iloc[0] + test_df.iloc[1]

        # Create a mask to filter out rows where either valence or test denominator is zero
        combined_mask = (valence_denominator != 0) & (test_denominator != 0)

        # Filter the DataFrames using the mask
        filtered_valence_df = valence_df.loc[:, combined_mask]
        filtered_test_df = test_df.loc[:, combined_mask]

        # Calculate initial valence and end valence
        initial_val = (filtered_valence_df.iloc[0] - filtered_valence_df.iloc[1]) / (filtered_valence_df.iloc[1] + filtered_valence_df.iloc[0])
        end_valence = (filtered_test_df.iloc[1] - filtered_test_df.iloc[0]) / (filtered_test_df.iloc[0] + filtered_test_df.iloc[1])

        # Calculate the learned index
        learned_index = (end_valence - initial_val) / 2

        # Return the learned index as a Series or DataFrame for appending to results
        return learned_index


def analyze_experiment_folder(folder_path, threshold, control_groups, experimental_groups):
    """
    This function analyzes all trials, organizes results by genotype, and outputs a CSV file with final results. 
    It also creates and saves a figure with bar plots, box plots, and swirl plots.
    """
    # Step 1: Initialize an empty DataFrame
    all_trials_data = pd.DataFrame()

    # Step 2: Traverse folder structure and collect data
    all_trials_data = collect_trial_data(folder_path, all_trials_data, threshold)

    # Step 3: Clean the DataFrame
    all_trials_data_cleaned = clean_trial_data(all_trials_data)

    # Step 4: Perform statistical analysis
    stats_results = perform_statistical_analysis(all_trials_data_cleaned, control_groups, experimental_groups)

    # Step 5: Save the cleaned data and stats results to CSV
    save_results_to_csv(folder_path, all_trials_data_cleaned, stats_results)

    # Step 6: Create and save the plots with significance markers
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
                        # Load metadata
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        fly_genotype = metadata.get('flyGenotype')

                        # Initialize MultiplexTrial object and analyze the trial
                        trial = MultiplexTrial()
                        trial.load_data(data_path)
                        trial.filter_by_num_choices(midline_borders=0.6, threshold=threshold, filter='both')
                        learned_index = trial.analyse_time()

                        # Convert trial's learned index to DataFrame and append
                        trial_data = pd.DataFrame({fly_genotype: learned_index})
                        all_trials_data = pd.concat([all_trials_data, trial_data], ignore_index=True)
    
    return all_trials_data

def clean_trial_data(all_trials_data):
    # Remove NaN values and reset indices
    return all_trials_data.apply(lambda x: x.dropna().reset_index(drop=True))

def perform_statistical_analysis(all_trials_data, control_groups, experimental_groups):
    """
    Perform statistical analysis between the experimental group(s) and each control group.
    Output the normality and variance check results along with the final statistical test result.
    Handles multiple comparisons and selects the appropriate test based on assumptions.
    """
    stats_results = []
    num_comparisons = len(control_groups) * len(experimental_groups)  # For Bonferroni correction

    for control in control_groups:
        for exp in experimental_groups:
            # Ensure data is numeric and drop NaNs
            control_data = pd.to_numeric(all_trials_data[control], errors='coerce').dropna()
            experimental_data = pd.to_numeric(all_trials_data[exp], errors='coerce').dropna()

            # Step 1: Check normality (Shapiro-Wilk test)
            control_normality_p = stats.shapiro(control_data).pvalue
            experimental_normality_p = stats.shapiro(experimental_data).pvalue
            normality_pass = control_normality_p > 0.05 and experimental_normality_p > 0.05

            # Step 2: Check equal variance (Levene's test)
            levene_p = stats.levene(control_data, experimental_data).pvalue
            equal_variance_pass = levene_p > 0.05

            # Step 3: Select the appropriate test
            if normality_pass and equal_variance_pass:
                # Use One-Way ANOVA since assumptions are met
                data = pd.concat([control_data, experimental_data], axis=0)
                group_labels = [control] * len(control_data) + [exp] * len(experimental_data)

                # Fit the ANOVA model
                model = ols('data ~ group_labels', data=pd.DataFrame({"data": data, "group_labels": group_labels})).fit()
                anova_result = sm.stats.anova_lm(model, typ=2)

                # If ANOVA is significant, run post-hoc Tukey's test
                if anova_result['PR(>F)'].iloc[0] < 0.05 / num_comparisons:  # Bonferroni correction
                    posthoc = pairwise_tukeyhsd(endog=data, groups=group_labels, alpha=0.05 / num_comparisons)
                    test_type = "ANOVA + Tukey's HSD"
                    test_statistic = anova_result['F'].iloc[0]
                    p_value = anova_result['PR(>F)'].iloc[0]
                else:
                    test_type = "ANOVA (not significant)"
                    test_statistic = anova_result['F'].iloc[0]
                    p_value = anova_result['PR(>F)'].iloc[0]

            else:
                # Use Kruskal-Wallis test if normality or equal variance fails
                kruskal_result = kruskal(control_data, experimental_data)
                test_type = "Kruskal-Wallis"
                test_statistic = kruskal_result.statistic
                p_value = kruskal_result.pvalue

                # If Kruskal-Wallis is significant, run post-hoc Mann-Whitney U test with Bonferroni correction
                if p_value < 0.05 / num_comparisons:
                    mannwhitney_result = stats.mannwhitneyu(control_data, experimental_data, alternative='two-sided')
                    posthoc_test_type = "Mann-Whitney U Test"
                    posthoc_statistic = mannwhitney_result.statistic
                    posthoc_p_value = mannwhitney_result.pvalue
                    p_value = posthoc_p_value
                    test_type += f" + {posthoc_test_type}"

            # Append results for this comparison with detailed checks
            stats_results.append({
                'Comparison': f'{exp} vs {control}',
                'Control Normality (p-value)': control_normality_p,
                'Experimental Normality (p-value)': experimental_normality_p,
                'Normality Assumption Pass': normality_pass,
                'Levene’s Test (p-value)': levene_p,
                'Equal Variance Assumption Pass': equal_variance_pass,
                'Test Type': test_type,
                'Test Statistic': test_statistic,
                'Test p-value': p_value
            })

    # Create a DataFrame for stats results
    stats_results_df = pd.DataFrame(stats_results)
    return stats_results_df



def save_results_to_csv(folder_path, all_trials_data_cleaned, stats_results):
    # Create output folder if it doesn't exist
    output_folder = os.path.join(folder_path, 'output')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save the cleaned data
    output_csv_path = os.path.join(output_folder, 'experiment_results_cleaned.csv')
    all_trials_data_cleaned.to_csv(output_csv_path, index=False)
    print(f"Cleaned results saved to {output_csv_path}")

    # Save the statistical results
    output_stats_path = os.path.join(output_folder, 'statistical_results.csv')
    stats_results.to_csv(output_stats_path, index=False)
    print(f"Statistical results saved to {output_stats_path}")

def create_and_save_plots(folder_path, all_trials_data_cleaned, stats_results):
    """
    Creates and saves a combined figure with bar plot, box plot, and swirl plot, 
    and also saves each plot individually in a folder.
    """
    # Reshape data for plotting
    all_trials_data_long = all_trials_data_cleaned.melt(var_name='Genotype', value_name='Learned Index')

    # Create a folder for the plots
    output_folder = os.path.join(folder_path, 'output', 'plots')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Combined Figure
    combined_fig_path = os.path.join(output_folder, 'combined_experiment_plots.png')
    plt.figure(figsize=(20, 10))

    # Bar Plot
    plt.subplot(1, 3, 1)  # Now adding plt.subplot here for combined figure
    create_bar_plot(all_trials_data_long, stats_results)

    # Box Plot
    plt.subplot(1, 3, 2)  # Define subplot for box plot in the combined figure
    create_box_plot(all_trials_data_long)

    # Swirl Plot
    plt.subplot(1, 3, 3)  # Define subplot for swirl plot in the combined figure
    create_swirl_plot(all_trials_data_long)

    # Save the combined figure
    plt.tight_layout()
    plt.savefig(combined_fig_path, dpi=300)
    print(f"Combined plots saved to {combined_fig_path}")
    plt.clf()  # Clear the figure for further plots

    # Save each individual plot
    save_individual_plots(output_folder, all_trials_data_long, stats_results)


def save_individual_plots(output_folder, data, stats_results):
    """
    Saves each individual plot (bar plot, box plot, swirl plot) in separate image files.
    """

    # Save Bar Plot
    bar_plot_path = os.path.join(output_folder, 'bar_plot.png')
    plt.figure(figsize=(6, 5))
    create_bar_plot(data, stats_results)
    plt.savefig(bar_plot_path, dpi=300)
    print(f"Bar plot saved to {bar_plot_path}")
    plt.clf()  # Clear the figure

    # Save Box Plot
    box_plot_path = os.path.join(output_folder, 'box_plot.png')
    plt.figure(figsize=(6, 5))
    create_box_plot(data)
    plt.savefig(box_plot_path, dpi=300)
    print(f"Box plot saved to {box_plot_path}")
    plt.clf()  # Clear the figure

    # Save Swirl Plot
    swirl_plot_path = os.path.join(output_folder, 'swirl_plot.png')
    plt.figure(figsize=(6, 5))
    create_swirl_plot(data)
    plt.savefig(swirl_plot_path, dpi=300)
    print(f"Swirl plot saved to {swirl_plot_path}")
    plt.clf()  # Clear the figure


def create_bar_plot(data, stats_results):
    """
    Creates a bar plot. Removed plt.subplot so it's flexible for individual and combined plots.
    """
    # Plot the error bars first with a lower zorder
    sns.barplot(data=data, x='Genotype', y='Learned Index', hue='Genotype', 
                palette="deep", errorbar="se", estimator="mean", capsize=0.1, zorder=5)

    # Plot the mean bars again without error bars with a higher zorder
    sns.barplot(data=data, x='Genotype', y='Learned Index', hue='Genotype', 
                palette="deep", errorbar=None, estimator="mean", edgecolor=".2", zorder=10)

    # Annotate significance
    add_statistical_annotations(stats_results)

    plt.ylim(-1, 1)
    plt.title('Bar Plot of Learned Index')
    plt.ylabel('Learned Index')
    plt.xlabel('Genotype')

def create_box_plot(data):
    """
    Creates a box plot. Removed plt.subplot so it's flexible for individual and combined plots.
    """
    sns.boxplot(data=data, x='Genotype', y='Learned Index', palette="deep", hue='Genotype')
    plt.ylim(-1, 1)
    plt.title('Box Plot of Learned Index')
    plt.ylabel('Learned Index')
    plt.xlabel('Genotype')

def create_swirl_plot(data):
    """
    Creates a swirl plot (swarm plot). Adjusted to center the points in each group.
    """
    sns.swarmplot(data=data, x='Genotype', y='Learned Index', palette="deep", hue="Genotype")
    plt.ylim(-1, 1)
    plt.title('Swirl Plot of Learned Index')
    plt.ylabel('Learned Index')
    plt.xlabel('Genotype')

def add_statistical_annotations(stats_results):
    """
    Add asterisks to indicate statistical significance on the bar plot.
    """
    significance_threshold = 0.05

    # Loop through each comparison result to add annotations
    for index, row in stats_results.iterrows():
        p_value = row['Test p-value']  # Updated to match the correct key name
        comparison = row['Comparison']

        # Split the comparison to get the group indices
        groups = comparison.split(" vs ")
        control_index = int(groups[1][-1]) - 1  # Extract the group number
        exp_index = int(groups[0][-1]) - 1

        # Add annotation if significant
        if p_value < significance_threshold:
            plt.text((control_index + exp_index) / 2, 0.9, '*', ha='center', va='bottom', color='black', fontsize=20)


# Example Usage:
analyze_experiment_folder(
    folder_path='G:/My Drive/Work/PhD Neuroscience/Moshe Parnas/Experiments/Serotonergic system/5ht_behavior/raw_data/multiplex/5ht_receptors_knockdown_classical/5ht1a_rnai', 
    threshold=4, 
    control_groups=['w1118x33885', 'w1118xmb247'], 
    experimental_groups=['33885xmb247']
)
