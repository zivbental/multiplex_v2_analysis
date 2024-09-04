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
        This function returns the period that corresponds to the initial valence of the fly to the two odors
        """
        # Load the MultiplexTrial object data into a temporary dataframe that will be manipulated
        df = self.processed_data
        initial_valence_df = df[(df['oct_left_status'] == 1) & (df['mch_right_status'] == 1)]
        
        # Fixed the regex pattern and used raw string notation to avoid escape sequence issues
        initial_valence_df = initial_valence_df.filter(regex=r'^(Timestamp|chamber_\d+_loc)$')
        
        # Set 'Timestamp' as the index
        initial_valence_df.set_index('Timestamp', inplace=True)

        return initial_valence_df

    def select_inital_valence_period(self):
        """
        This function returns the period that corresponds to the test period of the trial.
        """
        df = self.processed_data
        test_df = df[(df['mch_left_status'] == 1) & (df['oct_right_status'] == 1)]
        test_df = test_df.filter(regex=r'^(Timestamp|chamber_\d+_loc)$')
        test_df.set_index('Timestamp', inplace=True)
        return test_df

    def filter_by_num_choices(self, area_size, limit_range, threshold=1, filter='both'):
        """
        This function counts the number of times a fly has entered a center area.
        Values will indicate how well the fly has explored the chamber during the initial valence/test period.
        """
        valence_df = self.select_inital_valence_period()
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
            filtered_dfs[f"filtered_{key}"] = self.filter_by_midline(df, area_size=area_size, limit_range=limit_range, threshold=threshold)

        # If 'both' is selected, find the common columns between the two filtered DataFrames
        if filter == 'both':
            common_columns = filtered_dfs['filtered_valence_df'].columns.intersection(filtered_dfs['filtered_test_df'].columns)
            self.processed_data = filtered_dfs['filtered_valence_df'][common_columns], filtered_dfs['filtered_test_df'][common_columns]

        # Otherwise, return the filtered DataFrame itself
        elif filter in ['test', 'valence']:
            key = f"filtered_{filter}_df"
            self.processed_data = filtered_dfs[key][filtered_dfs[key].columns]


    def filter_by_midline(self, df, area_size, limit_range, threshold = 1):
        # Define the area limits
        center_size = area_size
        right_limit = 0 + center_size/2
        left_limit = 0 - center_size/2
        # Create a dataframe that will highlight when a fly crosses the limits, thereby creating 'chunks'.
        crosses_df = df[(((df >= right_limit - limit_range) & (df <= right_limit + limit_range)) | ((df <= left_limit + limit_range) & (df >= left_limit - limit_range)))]
        
        """
        The following code segment will count how many times the fly has crossed the limit. Because we get a lot of values for each limit cross,
        We combine the values we get into 'chunks', and count the number of chunks, which indicates the number of crosses.
        """

        # Shift the DataFrame down by 1 position
        shifted_df = crosses_df.shift(1)

        # Compare the original DataFrame to the shifted one
        # Identify where new NaNs appear in the shifted DataFrame compared to the original
        chunk_borders = (crosses_df.notna() & shifted_df.isna())

        # Count the number of chunk borders for each column
        chunk_border_count = chunk_borders.sum()

        columns_to_keep = chunk_border_count[chunk_border_count >= threshold].index
        filtered_df = df[columns_to_keep]
        
        return filtered_df
    
    @staticmethod
    def time_spent(df, width_size=20, sampling_rate=0.1):
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

        print(valence_df)
        print(test_df)
        print(f'valence average right: {valence_df.iloc[0].mean()}')
        print(f'valence average left: {valence_df.iloc[1].mean()}')
        print(f'test average right: {test_df.iloc[0].mean()}')
        print(f'test average left: {test_df.iloc[1].mean()}')

        # Calculate initial valence (left side [MCH] - right side [3-OCT]):
        initial_val = (valence_df.iloc[1] - valence_df.iloc[0]) / (valence_df.iloc[1] + valence_df.iloc[0])
        end_valence = (test_df.iloc[0] - test_df.iloc[1]) / (test_df.iloc[0] + test_df.iloc[1])

        # print(initial_val)
        # print(end_valence)

        # learned_index = (end_valence - initial_val) / 2

        # print(learned_index)
        # print(learned_index.mean())

    

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
# file_path = "/Users/zivbentulila/Library/CloudStorage/GoogleDrive-zivbental@gmail.com/My Drive/Work/MSc Neuroscience/Moshe Parnas/Experiments/Serotonergic system/5ht_behavior/operant_conditioning/raw_data/behavior/13.8.24/5ht_rnai/mb247/20240813_075645_Log.txt"
file_path = "C:/Users/user/Documents/Results/testing/4.9.24/mb320c/trial_1.csv"

# Create a MultiplexTrial object
trial_1 = MultiplexTrial()

# Load a single Trial to the object
trial_1.load_data(file_path)

trial_1.filter_by_num_choices(0.5, 0.3, 0, filter='both')

trial_1.analyse_time()



'''
Filtrations:
Master filter function to call various other functions:
- Filter by movement
- Filter by number of choises
- Filter out specific flies that I manually specify

Things I want to show:
Allow to select folder for analysis
Single fly trace (+ odor & shock timing)
All fly traces (+ odor & shock timing)
'''