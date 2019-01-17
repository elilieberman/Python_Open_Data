# Python_Open_Data
Python code for NYC Open Data Sources, tidy, features, charts

* Script to generate chart of New York City ride-share major competitors market-share 
Uses SODA api to download updated NYC open data for taxi industry.  
Pre-processing of strings fields to date objects, integers, for grouping operations.
Wrote function to tidy company names and multiple names for Uber, Lyft, VIA, Gett
Aggregated volume by month and presented in graph


* Script finding best Predictive Features of Restaurant Closing using NYC Open Data, Restaurant Violations
* Tidy the data, and learn if certain types of restaurants and violations, had a high correlation to closure, why? 
* Classification logistic regression,  which types of food (cuisine type) and violation types, may be leading indicators of restaurant closings (for health violations).   
* Preprocessed string fields, to time-objects, time-deltas, created dummies for cuisine and violation types, 
* Used kbest algrithm to find the most informative features.

# NYC TAP Files
Review of NYC open data related to metals concentration in NYC  tap water.
1. Coded in Python, primarily utilizing Pandas, Matplotlib and Basemap 
2. Intensity of map color parallels zip code areas relative concentration level (charts maps the copper concentrations in the 1-2 minute draw).
3. Data cleansing was performed on NYC open data, i.e. fixing data types, missing valuesnaming inconsistencies
4. Geocoding (latitiude and longitude) data was appended to the NYC dataset to facilitate the heat-map chart
