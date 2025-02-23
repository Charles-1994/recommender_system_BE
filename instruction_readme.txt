Implement all methods in the Recommender class found in recommender.py. The class will take 3 csv file paths in its constructor. The files will contain rows of items, users, events.

__init__(items_path, users_path, events_path) 
Read the contents of the files and store them in memory

train()
Train an internal prediction model, which will be used inside the recommend method

analyse()

Considering two sessions of the same user separate if there is a gap of >= 8 hours between two events, remove duplicate visits of the same item within the session(keep only the first) and excluding sessions with only a single event, output the following information about the dataset:
- Number of sessions
- Average (mean) number of events per session
- Histogram of session lengths in 10 bins
- Category with highest bounce rate (most often occurring as the last event within a session).
- Country with highest average visit count per user
- Performer with most visits by each category


recommend(session_item_ids)
Given a list of itemIds visited by a user within a session, recommend a list of items to the user with the goal of maximizing probability of actual interaction. Return a list of 5 itemIds.
For example for input [1,4,3,2,6,88,72], output: [3,8,2,12,1]



For evaluation evaluate.py will be executed instantiating the Recommender class and calling the analyse, train and recommend methods. 

Machine learning or recommender system libraries are not allowed to be used, but data manipulation tools are allowed and encouraged (ex. numpy, pandas).