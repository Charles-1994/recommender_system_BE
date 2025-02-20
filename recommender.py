import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

class Recommender:
    def __init__(self, items_path, users_path, events_path):
        #YOUR CODE HERE
        """Load all datasets from the datasets folder"""
        self.logger = self.configure_logging()
        self.logger.info("Loading datasets into pandas dataframes...")
        self.users_df = pd.read_csv(users_path)
        self.items_df = pd.read_csv(items_path)
        self.events_df = pd.read_csv(events_path)
        self.logger.info("Datasets loaded successfully")
        
    def analyse(self):
        #YOUR CODE HERE
        """Analyse the data"""
        self.logger.info("Analyzing data...")

        self.logger.info("Cleaning data...")
        cleaned_events = self.process_sessions(self.events_df)
        self.logger.info("Data cleaned successfully")

        self.logger.info(f"1. Total number of sessions with multiple events in a session: {len(cleaned_events.groupby('session_id'))}")

        self.logger.info(f"2. Average number of events per session: {cleaned_events.groupby('session_id')['item_id'].count().mean().round(2)}")

        # Calculate session length (in minutes)
        session_lengths = (cleaned_events.groupby('session_id')['datetime']
                  .agg(['min', 'max'])
                  .apply(lambda x: (x['max'] - x['min']).total_seconds() / 60, axis=1))
        self.logger.info("3. Plotting histogram of session lengths in minutes...")
        self.plot_histogram(session_lengths)

        # Calculate session length (in hours)
        session_lengths = (cleaned_events.groupby('session_id')['datetime']
                  .agg(['min', 'max'])
                  .apply(lambda x: (x['max'] - x['min']).total_seconds() / 3600, axis=1))
        self.logger.info("Plotting histogram of session lengths in hours...")
        self.plot_histogram(session_lengths, xlabel = "Session Length (hours)", title = "Distribution of Session Lengths in Hours")

        self.logger.info("4. Calculating the category with highest bounce rate...")
        bounce_items = cleaned_events.loc[(cleaned_events["bounce_flag"] == 1),'item_id'].value_counts().reset_index(name="N_bounces")
        bounce_cate = bounce_items.merge(self.items_df, on="item_id").groupby('category')['N_bounces'].sum().reset_index(name="Total_N_bounces")
        bounce_cate = bounce_cate.sort_values("Total_N_bounces", ascending=False).reset_index(drop=True)
        self.logger.info(f"The category with highest bounce rate is: {bounce_cate.iloc[0]['category']}")

        self.logger.info("5. Country with highest average visits per user...")
        user_visits = cleaned_events.groupby('user_id')['session_counter'].max().reset_index(name="n_visits")
        avg_visit_by_country = user_visits.merge(self.users_df, on="user_id").groupby("country")['n_visits'].mean().reset_index(name = 'avg_n_visits')
        avg_visit_by_country = avg_visit_by_country.sort_values("avg_n_visits", ascending=False).reset_index(drop=True)
        self.logger.info(f"The country with highest average visits per user is: {avg_visit_by_country.iloc[0]['country']}")

        self.logger.info("6. Performer with most number of visits by each category...")
        top_performers = cleaned_events.merge(self.items_df, on="item_id").groupby(['category','item_id'])['item_id'].count().reset_index(name="N_visits")
        top_performers = top_performers.sort_values('N_visits', ascending=False).groupby('category').head(1).reset_index(drop=True)
        self.logger.info(f"The performer with most number of visits by each category is: ")
        print(top_performers)

        self.logger.info("Analysis completed successfully")
    
    def train(self):
        #YOUR CODE HERE
        pass
        
    def recommend(self, session_items):
        #YOUR CODE HERE
        return []
    
    def process_sessions(self, events):
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
        """Remove duplicates items with the same session keeping the first records by time"""
        self.logger.info("Removing duplicates...")
        df = (df.sort_values(['user_id', 'session_id', 'timestamp', 'item_id'])
            .drop_duplicates(subset=['user_id', 'session_id', 'item_id'], keep='first')
            .reset_index(drop=True))
        return df

    def filter_multievent_sessions(self, df):
        """Filter out sessions with only one event"""
        self.logger.info("Filtering multievent sessions...")
        # Count events per session
        session_counts = df.groupby(['user_id', 'session_id']).size().reset_index(name='event_count')

        # Filter out sessions with only one event
        df = df.merge(session_counts, on=['user_id', 'session_id'])
        df = df[df['event_count'] > 1].drop(columns=['event_count'])
        
        return df
    
    def plot_histogram(self, data, xlabel = "Session Length (minutes)", ylabel = "Frequency",title = "Distribution of Session Lengths"):
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
       
    def configure_logging(self):
        """Configure logging"""

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Create logger
        return logging.getLogger('recommender')

