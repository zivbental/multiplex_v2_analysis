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


def analyze_experiment_folder(folder_path, threshold):
    """
    This function analyzes all the trials in a given folder, organizes the results by genotype, and outputs
    a CSV file with the final results. It also creates and saves a figure with bar plots, box plots, and swirl plots.
    The output will be stored in a subfolder named 'output' inside the provided folder path.
    """
    # Step 1: Initialize an empty DataFrame to hold all trial data
    all_trials_data = pd.DataFrame()

    # Step 2: Traverse the folder structure
    for date_folder in os.listdir(folder_path):
        date_path = os.path.join(folder_path, date_folder)

        if os.path.isdir(date_path):
            for trial_folder in os.listdir(date_path):
                trial_path = os.path.join(date_path, trial_folder)

                if os.path.isdir(trial_path):
                    # Load metadata and CSV data
                    metadata_path = os.path.join(trial_path, 'experiment_metadata.json')
                    data_path = os.path.join(trial_path, 'fly_loc.csv')

                    if os.path.exists(metadata_path) and os.path.exists(data_path):
                        # Read the metadata file
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        fly_genotype = metadata.get('flyGenotype')

                        # Initialize MultiplexTrial object and analyze the trial
                        trial = MultiplexTrial()
                        trial.load_data(data_path)
                        trial.filter_by_num_choices(midline_borders=0.6, threshold=threshold, filter='both')
                        learned_index = trial.analyse_time()

                        # Convert the trial's learned index into a DataFrame for appending
                        trial_data = pd.DataFrame({fly_genotype: learned_index})

                        # Append the trial data to the main DataFrame
                        all_trials_data = pd.concat([all_trials_data, trial_data], ignore_index=True)

    # Step 3: Remove empty values (NaN) from each column
    all_trials_data_cleaned = all_trials_data.apply(lambda x: x.dropna().reset_index(drop=True))

    # Step 4: Create an output folder within the folder path
    output_folder = os.path.join(folder_path, 'output')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Step 5: Output the final DataFrame to a CSV file in the output folder
    output_csv_path = os.path.join(output_folder, 'experiment_results_cleaned.csv')
    all_trials_data_cleaned.to_csv(output_csv_path, index=False)
    print(f"Cleaned results saved to {output_csv_path}")

    # Step 6: Reshape the data for plotting (melt the DataFrame to long format)
    all_trials_data_long = all_trials_data_cleaned.melt(var_name='Genotype', value_name='Learned Index')

    # Step 7: Generate the bar plot, box plot, and swirl plot
    plt.figure(figsize=(20, 10))

    # Bar Plot
    plt.subplot(1, 3, 1)
    sns.barplot(data=all_trials_data_long, x='Genotype', y='Learned Index')
    plt.ylim(-1, 1)
    plt.title('Bar Plot of Learned Index')
    plt.ylabel('Learned Index')
    plt.xlabel('Genotype')

    # Box Plot
    plt.subplot(1, 3, 2)
    sns.boxplot(data=all_trials_data_long, x='Genotype', y='Learned Index')
    plt.ylim(-1, 1)
    plt.title('Box Plot of Learned Index')
    plt.ylabel('Learned Index')
    plt.xlabel('Genotype')

    # Swirl Plot (Strip plot with jitter to show individual data points)
    plt.subplot(1, 3, 3)
    sns.swarmplot(data=all_trials_data_long, x='Genotype', y='Learned Index', dodge=True)
    plt.ylim(-1, 1)
    plt.title('Swirl Plot of Learned Index')
    plt.ylabel('Learned Index')
    plt.xlabel('Genotype')

    # Adjust layout and save the figure to the output folder
    plt.tight_layout()
    output_fig_path = os.path.join(output_folder, 'experiment_plots.png')
    plt.savefig(output_fig_path, dpi=300)
    print(f"Plots saved to {output_fig_path}")

# Example Usage:
analyze_experiment_folder('C:/Users/user/Documents/Results/Ziv/5ht_receptors_knockdown_operant/5ht1a_rnai', threshold=1)