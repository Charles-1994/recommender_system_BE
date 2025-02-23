import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from itertools import combinations
from collections import defaultdict

class Recommender:
    def __init__(self, items_path, users_path, events_path):
        #YOUR CODE HERE
        """
        Initialize the Recommender system by loading datasets and configuring logging.

        Args:
            items_path (str): Path to the items dataset.
            users_path (str): Path to the users dataset.
            events_path (str): Path to the events dataset.
        """
        self.logger = self.configure_logging()
        self.logger.info("Loading datasets into pandas dataframes...")
        self.users_df = pd.read_csv(users_path)
        self.items_df = pd.read_csv(items_path)
        self.events_df = pd.read_csv(events_path)
        self.similarity_matrix = None
        self.cleaned_events = None
        self.logger.info("Datasets loaded successfully")
        
    def analyse(self):
        #YOUR CODE HERE
        """
        Perform analysis on the events data to extract insights such as session statistics,
        bounce rates, and top performers by category.
        """
        self.logger.info("Analyzing data...")

        self.logger.info("Cleaning data...")
        self.cleaned_events = self.process_sessions(self.events_df)
        self.logger.info("Data cleaned successfully")

        self.logger.info(f"1. Total number of sessions with multiple events in a session: {len(self.cleaned_events.groupby('session_id'))}")

        self.logger.info(f"2. Average number of events per session: {self.cleaned_events.groupby('session_id')['item_id'].count().mean().round(2)}")

        # Calculate session length (in minutes)
        session_lengths = (self.cleaned_events.groupby('session_id')['datetime']
                  .agg(['min', 'max'])
                  .apply(lambda x: (x['max'] - x['min']).total_seconds() / 60, axis=1))
        self.logger.info("3. Plotting histogram of session lengths in minutes...")
        self.plot_histogram(session_lengths)

        # Calculate session length (in hours)
        session_lengths = (self.cleaned_events.groupby('session_id')['datetime']
                  .agg(['min', 'max'])
                  .apply(lambda x: (x['max'] - x['min']).total_seconds() / 3600, axis=1))
        self.logger.info("Plotting histogram of session lengths in hours...")
        self.plot_histogram(session_lengths, xlabel = "Session Length (hours)", title = "Distribution of Session Lengths in Hours")

        self.logger.info("4. Calculating the category with highest bounce rate...")
        bounce_items = self.cleaned_events.loc[(self.cleaned_events["bounce_flag"] == 1),'item_id'].value_counts().reset_index(name="N_bounces")
        bounce_cate = bounce_items.merge(self.items_df, on="item_id").groupby('category')['N_bounces'].sum().reset_index(name="Total_N_bounces")
        bounce_cate = bounce_cate.sort_values("Total_N_bounces", ascending=False).reset_index(drop=True)
        self.logger.info(f"The category with highest bounce rate is: {bounce_cate.iloc[0]['category']}")

        self.logger.info("5. Country with highest average visits per user...")
        user_visits = self.cleaned_events.groupby('user_id')['session_counter'].max().reset_index(name="n_visits")
        avg_visit_by_country = user_visits.merge(self.users_df, on="user_id").groupby("country")['n_visits'].mean().reset_index(name = 'avg_n_visits')
        avg_visit_by_country = avg_visit_by_country.sort_values("avg_n_visits", ascending=False).reset_index(drop=True)
        self.logger.info(f"The country with highest average visits per user is: {avg_visit_by_country.iloc[0]['country']}")

        self.logger.info("6. Performer with most number of visits by each category...")
        top_performers = self.cleaned_events.merge(self.items_df, on="item_id").groupby(['category','item_id'])['item_id'].count().reset_index(name="N_visits")
        top_performers = top_performers.sort_values('N_visits', ascending=False).groupby('category').head(1).reset_index(drop=True)
        self.logger.info(f"The performer with most number of visits by each category is: ")
        print(top_performers)

        self.logger.info("Analysis completed successfully")
    
    def train(self, weighted=True):
        #YOUR CODE HERE
        """
        Train the recommender system by calculating a similarity matrix based on co-occurrence.

        Args:
            weighted (bool): If True, calculate a weighted similarity matrix using item weights.
                             If False, calculate a normal similarity matrix.

        Returns:
            None: The similarity matrix is stored as an instance variable.
        """
        self.logger.info("Training the recommender system...")
        # Step 1: Prepare session data
        session_train = self.cleaned_events.groupby('session_id')['item_id'].apply(list)

        # Initialize co-occurrence matrix
        co_occurrence = defaultdict(lambda: defaultdict(float)) if weighted else defaultdict(lambda: defaultdict(int))

        if weighted:
            self.logger.info("Calculating weighted similarity matrix...")
            # Step 1: Calculate N_visits per item
            item_visits = self.cleaned_events['item_id'].value_counts().reset_index(name='N_visits')
            item_visits.columns = ['item_id', 'N_visits']

            # Step 2: Merge with items dataset to get category and N_visits
            items_with_weights = self.items_df.merge(item_visits, on='item_id', how='left')

            # Step 3: Create weight mapping (example using N_visits or category weights)
            category_weights = {'dance': 1.2, 'flirt': 1.0, 'dj': 1.4}
            items_with_weights['weight'] = items_with_weights['category'].map(category_weights)

            # Step 4: Create a weight dictionary for quick lookup
            weight_dict = dict(zip(items_with_weights['item_id'], items_with_weights['weight']))

            # Step 5: Populate weighted co-occurrence matrix
            for items_in_session in session_train:
                unique_items = set(items_in_session)
                for item1, item2 in combinations(unique_items, 2):
                    # Get weights for both items
                    weight1 = weight_dict.get(item1, 1.0)  # Default to 1.0 if weight not found
                    weight2 = weight_dict.get(item2, 1.0)

                    # Calculate weighted co-occurrence
                    weighted_count = (weight1 + weight2) / 2  # Average of both weights

                    # Update co-occurrence matrix
                    co_occurrence[item1][item2] += weighted_count
                    co_occurrence[item2][item1] += weighted_count

        else:
            self.logger.info("Calculating similarity matrix...")
            # Populate normal co-occurrence matrix
            for items in session_train:
                for item1, item2 in combinations(set(items), 2):  # Get unique pairs of items
                    co_occurrence[item1][item2] += 1
                    co_occurrence[item2][item1] += 1

        # Step 6: Convert to DataFrame for easier manipulation
        co_occurrence_df = pd.DataFrame(co_occurrence).fillna(0)

        # Step 7: Normalize the similarity matrix (shared logic)
        self.similarity_matrix = self._normalize_similarity_matrix(co_occurrence_df)
        self.logger.info("Training completed successfully")
        
    def recommend(self, session_items,top_n=5):
        #YOUR CODE HERE
        """
        Recommend items based on input session items using the trained similarity matrix.

        This method calculates a score for each candidate item based on its similarity
        to the input items and returns the top N recommendations.

        Args:
            session_items (list): List of item IDs consumed by the user in the current session.
            top_n (int): Number of recommendations to return. Default is 5.

        Returns:
                list: A list of the top N recommended item IDs.
        """

        self.logger.info("Recommending items...")
        if self.similarity_matrix is None:
            raise ValueError("The model has not been trained yet. Please call train() first.")

        # Initialize scores dictionary
        scores = defaultdict(float)

        # For each candidate item (all items in the similarity matrix)
        for candidate_item in self.similarity_matrix.columns:
            if candidate_item not in session_items:  # Exclude input items
                # Calculate the total co-occurrence score with all input items
                total_score = sum(
                    self.similarity_matrix.loc[candidate_item, input_item]
                    for input_item in session_items if input_item in self.similarity_matrix.columns
                )
                scores[candidate_item] = total_score

        # Sort by aggregated scores and return top N recommendations
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        self.logger.info(f"top {top_n} recommendations: ")
        print([item for item, _ in sorted_scores[:top_n]])
        return [item for item, _ in sorted_scores[:top_n]]

    def process_sessions(self, events):
        """
        Process the events data to create session IDs and remove duplicates.

        This method converts the timestamp column to a datetime object, sorts the events by user_id and timestamp,
        calculates the time difference between consecutive events, identifies new sessions based on a time threshold,
        assigns session IDs, removes duplicate events, and filters out sessions with only one event.
        """

        self.logger.info("Processing sessions and creating session IDs...")

        self.logger.info("converting timestamp column to datetime...")
        # Convert timestamp to datetime
        events['datetime'] = pd.to_datetime(events['timestamp'], unit='s')
        # Sort by user_id and timestamp
        events = events.sort_values(by=["user_id", "timestamp"])

        # Calculate time difference between consecutive events
        events['hours_since_last'] = events.groupby('user_id')['datetime'].diff().dt.total_seconds() / 3600

        self.logger.info("Identifying new sessions (time_diff > 8 hours)...")
        # Identify new sessions (time_diff > 8 hours)
        events['new_session'] = (events['hours_since_last'] >= 8 ).fillna(True)

        # Assign session IDs
        events['session_counter'] = events.groupby('user_id')['new_session'].cumsum()
        events['session_id'] = events['user_id'].astype(str)+'-'+events['session_counter'].astype(str)

        events = self.remove_duplicates(events)
        events = self.filter_multievent_sessions(events)

        # Adding bounce flag column
        self.logger.info("Adding bounce flag column...")
        events['bounce_flag'] = (events.groupby('session_id')['datetime']
                                .transform('max') == events['datetime']).astype(int)

        return events
    
    def remove_duplicates(self, df):
        """
        Remove duplicates items with the same session keeping the first records by time
        Args:
            df (pd.DataFrame): The events data to remove duplicates from.

        Returns:
            pd.DataFrame: The events data with duplicates removed.
        """
        self.logger.info("Removing duplicates...")
        df = (df.sort_values(['user_id', 'session_id', 'timestamp', 'item_id'])
            .drop_duplicates(subset=['user_id', 'session_id', 'item_id'], keep='first')
            .reset_index(drop=True))
        return df

    def filter_multievent_sessions(self, df):
        """
        Filter out sessions with only one event
        Args:
            df (pd.DataFrame): The events data to filter.

        Returns:
            pd.DataFrame: The filtered events data.
        """
        self.logger.info("Filtering multievent sessions...")
        # Count events per session
        session_counts = df.groupby(['user_id', 'session_id']).size().reset_index(name='event_count')

        # Filter out sessions with only one event
        df = df.merge(session_counts, on=['user_id', 'session_id'])
        df = df[df['event_count'] > 1].drop(columns=['event_count'])
        
        return df
    
    def plot_histogram(self, data, xlabel = "Session Length (minutes)", ylabel = "Frequency",title = "Distribution of Session Lengths"):
        """
        Plot a histogram of the session lengths.

        Args:
            data (pd.Series): The session lengths to plot.
            xlabel (str): The label for the x-axis.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=10, edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)

        # Add mean and median lines
        plt.axvline(data.mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {data.mean():.2f} mins')
        plt.axvline(data.median(), color='green', linestyle='dashed', linewidth=1, label=f'Median: {data.median():.2f} mins')
        plt.legend()
        plt.show()

        # Show some basic statistics
        self.logger.info("Session Length Statistics:")
        print(data.describe())

    def _normalize_similarity_matrix(self, co_occurrence_df):
        """
        Normalize a co-occurrence matrix to compute similarity scores.
        
        Args:
            co_occurrence_df (pd.DataFrame): Co-occurrence matrix.
        
        Returns:
            pd.DataFrame: Normalized similarity matrix.
        """
        self.logger.info("Normalizing similarity matrix...")
        normalized_matrix = co_occurrence_df.copy()
        for item in normalized_matrix.columns:
            norm_factor = np.sqrt(normalized_matrix[item].sum())
            if norm_factor > 0:  # Avoid division by zero
                normalized_matrix[item] /= norm_factor
        return normalized_matrix
       
    def configure_logging(self):
        """Configure logging"""

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Create logger
        return logging.getLogger('recommender')

