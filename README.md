# Comparing Different Machine Learning Models to Predict Group Performance

Using different regression and classification machine learning models we compare how they do to each other while trying to predict the group performance from the GAP corpus dataset. 
To do this we use ConvoKit to get the datasets feature. 

# How To Run Files

To run files there is three steps.
- Choose which features you want to include in the dataframe. To do this you need to add features from the get_attributes.py file into the dataframe that is in the get_data_frame.py file.
- Run the get_data_frame.py file. This will get the dataframe and save it as a CSV file. From this CSV file you can test different machine learing models.
- Run the neural_network_models.py file to test how different machine learning models compare to one another. 
