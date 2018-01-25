# Python_Open_Data
Python code for NYC Open Data Sources, tidy, features, charts

* Script to generate chart of New York City ride-share major competitors market-share 
Uses SODA api to download updated NYC open data for taxi industry.  
Pre-processing of strings fields to date objects, integers, for grouping operations.
Wrote function to tidy company names and multiple names for Uber, Lyft, VIA, Gett
Aggregated volume by month and presented in graph


* NYC Open Data, Department of Health, Restaurant Violations 
* Script to tidy the data, and learn if certain types of restaurants and violations, had a high correlation to closure, why? 
Classification logistic regression,  which types of food (cuisine type) and violation types, may be leading indicators of restaurant closings (for health violations).   
Preprocesses string fields, to time-objects, time-deltas, created dummies for cuisine and violation types, used kbest algrithm to find the most informative features.
