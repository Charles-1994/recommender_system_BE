# **README: ByBorg Recommendation System**

## **Introduction**
The ByBorg Recommendation System is a Python-based project designed to analyze user interactions with performers and provide personalized recommendations. The system processes user interaction data, identifies behavioral patterns, and recommends items (performers) based on co-occurrence patterns. The goal is to maximize the probability of user engagement with recommended items.

This project involves analyzing session-based user interaction data, training a recommendation model, and generating recommendations using a co-occurrence-based approach. It also includes exploratory analysis to uncover key insights about user behavior, session dynamics, and performer popularity.

---

## **Project Overview**

### **Data Sets**
The project utilizes three datasets:
1. **Users Dataset (`users.csv`)**:
   - Contains user information (`user_id`, `country`).
   - 500 users from 7 countries.

2. **Events Dataset (`events.csv`)**:
   - Logs user interactions with performers (`user_id`, `event_id`, `timestamp`).
   - 16,077 events recorded with timestamps in UNIX format.

3. **Items Dataset (`items.csv`)**:
   - Represents performers with attributes like `item_id`, `category` (e.g., "dance", "flirt", "dj"), `hair`, and `eyes`.

---

## **Tasks**

### **1. Data Initialization**
- The class constructor (`__init__`) reads the datasets into memory as Pandas DataFrames:
  - `users_df`: User data.
  - `events_df`: Interaction data.
  - `items_df`: Performer data.

---

### **2. Data Analysis**
The `analyse()` method processes and analyzes the interaction data to extract insights about sessions, events, and user behavior:
1. **Session Processing**:
   - Sessions are differentiated based on an 8-hour inactivity threshold.
   - Each session is assigned a unique `session_id` (e.g., `user_id-1`, `user_id-2`).

2. **Data Cleaning**:
   - Duplicate items within the same session are removed (keeping the earliest event).
   - Sessions with only one event are filtered out.

3. **Key Metrics**:
   - Total number of sessions: **1,026**.
   - Average number of events per session: **14.69**.
   - Session lengths are calculated and visualized in a histogram (divided into 10 bins).

4. **Behavioral Insights**:
   - *Category with Highest Bounce Rate*: The last event in each session is considered a bounce event. The category with the highest bounce rate is identified.
   - *Country with Highest Average Visits*: A visit corresponds to a session. The country with the highest average visits per user is calculated.
   - *Top Performers by Category*: Performers with the most visits are identified for each category.

---

### **3. Model Training**
The `train()` method builds a co-occurrence-based similarity matrix to recommend items based on their interaction patterns:
- **Co-Occurrence Matrix**:
  - Tracks how often pairs of items are consumed together in sessions.
  - Rows and columns represent item IDs, and cell values represent co-occurrence counts.
- **Weighted Similarity Matrix**:
  - Adds weights to items based on their popularity (`N_visits`) and category-specific weights (`category_weights = {'dance': 1.2, 'flirt': 1.0, 'dj': 1.4}`).
- The similarity matrix is normalized to scale values for consistent scoring.

---

### **4. Recommendations**
The `recommend()` method generates recommendations for users based on their session history:
- Takes a list of item IDs visited by a user within a session as input.
- Calculates co-occurrence scores for all other items not in the input list.
- Returns the top 5 recommended item IDs based on similarity scores.

---

## **Recommendation Model**

### Approach
The recommendation model uses a co-occurrence-based approach to identify items frequently consumed together in sessions:
1. For each pair of items consumed in the same session, their co-occurrence count is incremented.
2. A similarity matrix is built using these counts, optionally weighted by item popularity and category importance.

### Evaluation
To evaluate the model's performance:
- The training dataset includes data up to December 24, 2020.
- Recommendations are tested on sessions after this date.
- Observations show that at least one recommendation matches the user's actual interaction in most cases.

---

## **Future Improvements**
1. Incorporate more advanced techniques like cosine similarity or Apriori algorithms for better combination detection between items.
2. Introduce personalized recommendations by incorporating user-specific behavior patterns.
3. Explore hybrid models combining content-based filtering (using item attributes) with collaborative filtering.

---

## **Key Features**
- Session-based analysis of user interactions.
- Co-occurrence-based recommendation model for high engagement probability.
- Insights into user behavior patterns and performer popularity.

This project demonstrates how simple yet effective techniques can be used to build recommendation systems without relying on external libraries or machine learning frameworks.
