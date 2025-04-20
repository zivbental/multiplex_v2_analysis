import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os



class FolderAnalysis:
    """
    This class is meant to collect all trials in a given folder and perform analysis on all of them.
    It will use the other class (MultiplexTrial), through which the various manipulations of the data will be performed.

    Therefore, the goal of this class is to orgnaize all the files in a given folder and present statistics,
    figures and data tables for the analysis of multiple trials.
    """
    def __init__(self) -> None:
        pass

    def create_datatables(self):
        pass

    def create_figures(self):
        pass

class MultiplexTrial:
    def __init__(self) -> None:
        raw_data = None
        processed_data = None

        pass

    def load_data(self, data_path):
        """
        This function loads a Multiplex log file (.txt) and coverts it to pandas dataframe
        """
        self.processed_data = pd.read_csv(data_path)

    def select_test_period(self):
        """
        This function returns the period that corresponds to the 'Test' phase of the trial.
        """
        # Load the MultiplexTrial object data into a temporary dataframe that will be manipulated
        df = self.processed_data
        
        # Select the rows where the 'experiment_step' column is 'Test'
        test_df = df[df['experiment_step'] == 'Test']
        
        # Filter the dataframe to include only Timestamp and chamber locations
        test_df = test_df.filter(regex=r'^(Timestamp|chamber_\d+_loc)$')
        
        # Set 'Timestamp' as the index
        test_df.set_index('Timestamp', inplace=True)

        return test_df

    def select_initial_valence_period(self):
        """
        This function returns the period that corresponds to the 'Initial Valence' phase of the trial.
        """
        # Load the MultiplexTrial object data into a temporary dataframe that will be manipulated
        df = self.processed_data
        
        # Select the rows where the 'experiment_step' column is 'Initial Valence'
        initial_valence_df = df[df['experiment_step'] == 'Initial Valence']
        
        # Filter the dataframe to include only Timestamp and chamber locations
        initial_valence_df = initial_valence_df.filter(regex=r'^(Timestamp|chamber_\d+_loc)$')
        
        # Set 'Timestamp' as the index
        initial_valence_df.set_index('Timestamp', inplace=True)

        return initial_valence_df

    def filter_by_num_choices(self, midline_borders, threshold=1, filter='both'):
        """
        This function counts the number of times a fly has entered a center area.
        Values will indicate how well the fly has explored the chamber during the initial valence/test period.
        """
        valence_df = self.select_initial_valence_period()
        test_df = self.select_test_period()

        # Create a mapping of filters to DataFrames
        df_mapping = {
            'both': [('valence_df', valence_df), ('test_df', test_df)],
            'test': [('test_df', test_df)],
            'valence': [('valence_df', valence_df)]
        }

        # Dictionary to store the filtered DataFrames
        filtered_dfs = {}

        # Apply filtering based on the filter key, if it exists in the mapping
        for key, df in df_mapping.get(filter, []):
            filtered_dfs[f"filtered_{key}"] = self.filter_by_midline(df, midline_borders=midline_borders, threshold=threshold)

        # If 'both' is selected, find the common columns between the two filtered DataFrames
        if filter == 'both':
            common_columns = filtered_dfs['filtered_valence_df'].columns.intersection(filtered_dfs['filtered_test_df'].columns)
            self.processed_data = filtered_dfs['filtered_valence_df'][common_columns], filtered_dfs['filtered_test_df'][common_columns]

        # Otherwise, return the filtered DataFrame itself
        elif filter in ['test', 'valence']:
            key = f"filtered_{filter}_df"
            self.processed_data = filtered_dfs[key][filtered_dfs[key].columns]


    def filter_by_midline(self, df, midline_borders, threshold = 1):
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
        
        # Filter out columns that have not crossed the threshold times
        filtered_columns = [col for col, count in crossing_counts.items() if count >= threshold]
        filtered_df = df[filtered_columns]
        
        return filtered_df
    
    @staticmethod
    def time_spent(df, width_size=10, sampling_rate=0.1):
        """
        This function determines which 'side' the fly is currently in based on the given width_size.
        It then calculates the time the fly spent on each side of the chamber.
        Returns a dataframe that shows for each fly the time spent on each side.
        """

        # Helper function to process mask results and return transposed dataframe
        def process_counts(counts):
            # Reset index and transpose
            df_transposed = counts.reset_index().T
            # Assign the first row as column names
            df_transposed.columns = df_transposed.iloc[0]
            # Drop the first row (now header) and return the modified DataFrame
            return df_transposed.drop(df_transposed.index[0])

        # Create boolean masks for values greater than width_size and less than -width_size
        mask_greater = df > width_size
        mask_less = df < -width_size

        # Calculate time spent based on sampling rate
        count_greater = mask_greater.sum() * sampling_rate
        count_less = mask_less.sum() * sampling_rate

        # Process the counts and create dataframes
        count_greater_processed = process_counts(count_greater)
        count_less_processed = process_counts(count_less)

        # Concatenate the processed dataframes
        df_combined = pd.concat([count_greater_processed, count_less_processed])

        # Assign index names 'right_side' and 'left_side'
        df_combined.index = ['right_side', 'left_side']

        return df_combined


    def analyse_time(self):

        valence_df = self.time_spent(self.processed_data[0])
        test_df = self.time_spent(self.processed_data[1])

        # Create denominators for valence_df and test_df
        valence_denominator = valence_df.iloc[1] + valence_df.iloc[0]
        test_denominator = test_df.iloc[0] + test_df.iloc[1]

        # Create a mask to filter out rows where either valence or test denominator is zero
        combined_mask = (valence_denominator != 0) & (test_denominator != 0)

        # Apply the mask to both valence_df and test_df
        filtered_valence_df = valence_df.loc[:, combined_mask]
        filtered_test_df = test_df.loc[:, combined_mask]

        # Filter out columns where initial_val (filtered_valence_df.iloc[1]) is less than 30
        initial_val_filter = filtered_valence_df.iloc[1] >= 30

        filtered_valence_df = filtered_valence_df.loc[:, initial_val_filter]
        filtered_test_df = filtered_test_df.loc[:, initial_val_filter]

        # Calculate initial valence and end valence using the filtered data
        # initial_val = (filtered_valence_df.iloc[1]) / 120
        # end_valence = (filtered_test_df.iloc[0]) / 120

        # Option B
        initial_val = (filtered_valence_df.iloc[1]) / (filtered_valence_df.iloc[0] + filtered_valence_df.iloc[1])
        end_valence = (filtered_test_df.iloc[0]) / (filtered_test_df.iloc[0] + filtered_test_df.iloc[1])

        # Calculate the learned index based on the filtered data
        learned_index = (end_valence - initial_val) * 100
        print("Time before")
        print(initial_val*100)
        print("Time After")
        print(end_valence*100)
        


        # Calculate the learned index based on the filtered data
        learned_index = (end_valence - initial_val)*100

        print("Learned Index")
        print(learned_index)
        print(learned_index.mean())

    

    def filter_specific_fly(self):
        pass

    
    def plot_trial(self):
        # Load the MultiplexTrial object data into a temporary dataframe that will be manipulated
        df = self.processed_data

        # Select only the test period (may vary if I will use other protocols)
        test_df = df[(df['LEFTODOR2'] == 1) & (df['RIGHTODOR1'] == 1)]

        # Extract time and location columns of flies
        test_df_location_only = test_df.filter(regex='^(Timestamp|cX\d{3})$')

        # Replace 0 values with NaN, to not include during the calculation 0, which happens when the system is not detectiing the flies
        test_df_location_only.replace(0, np.nan, inplace=True)

        # Set the 'Time' column as the index
        test_df_location_only.set_index('Time', inplace=True)
        # Plot data
        cX_columns = [col for col in test_df_location_only.columns if col.startswith('cX')]

        # Set up the figure and axis for subplots
        fig, axes = plt.subplots(len(cX_columns), 1, figsize=(10, 0.5 * len(cX_columns)), sharex=True)

        # If there's only one subplot, wrap the axes in a list for consistency
        if len(cX_columns) == 1:
            axes = [axes]

        # Plot each cX column in its own subplot
        for i, col in enumerate(cX_columns):
            sns.lineplot(ax=axes[i], x=test_df_location_only.index, y=test_df_location_only[col])
            axes[i].set_ylim(-1, 1)  # Set Y-axis limits from -1 to 1
            axes[i].set_ylabel('')  # Remove the Y-axis label
            axes[i].set_yticks([-1, 0, 1])  # Optionally, set specific y-ticks
            
            # Place the column name on the left side
            axes[i].annotate(col, xy=(0, 0.5), xytext=(-axes[i].yaxis.labelpad - 10, 0),
                             xycoords=axes[i].yaxis.label, textcoords='offset points',
                             size='large', ha='right', va='center', rotation=0)

            axes[i].grid(True)

        # Set the common x-label
        axes[-1].set_xlabel('Time')

        # Adjust the layout to prevent overlap
        plt.tight_layout()

        # Display the plot
        plt.show()


"""
Main segment of code that runs the functions
"""
file_path = "//132.66.94.156/Ziv/multiplex/Danielle/training/w1118_learning/26.03.2025/trial_2/fly_loc.csv"
# file_path = "C:/Users/user/Documents/Results/Dekel/MchRB_Voltage_independent/mchrb-VID/12.03.2025/trial_7/fly_loc.csv"

# Create a MultiplexTrial object
trial_1 = MultiplexTrial()

# Load a single Trial to the object
trial_1.load_data(file_path)

trial_1.filter_by_num_choices(midline_borders=0.6, threshold=4, filter='both')

trial_1.analyse_time()


