import streamlit as st
import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

# Hiding the menu and header when serving the app
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Loading the model and the indexer from the saved files
indexer = PipelineModel.load('airport-index/')
model = PipelineModel.load('airport-shiz/')

# Creating a Spark session
spark = SparkSession.builder.appName('airport').getOrCreate()

# Setting the page title 
html_string = '''
<h1><a style='text-decoration:none; color:white;' href='https://github.com/Amaaan09/pyspark-airport'>DelayDecoded</a></h1><h5><a style='text-decoration:none; color:white;' href='https://github.com/Amaaan09/pyspark-airport'>Click here to go to the GitHub repository</a></h5> 
'''
st.markdown(html_string, unsafe_allow_html=True)

# Loading the csv for the sidebar default values
df = pd.read_csv("airport-data.csv")

# Creating the sidebar
st.sidebar.title("Input Features")
st.sidebar.markdown("Select the features that you want to input.")

# Adding the sidebar inputs
airline = st.sidebar.selectbox("Airline", df["Airline"].unique())
origin = st.sidebar.selectbox("Origin", df["Origin"].unique())
dest = st.sidebar.selectbox("Destination", df["Dest"].unique())
distance = st.sidebar.number_input("Distance", value=df['Distance'].mean())
crs_arr_time = st.sidebar.number_input("CRS Arrival Time", value=df['CRSArrTime'].median())
crs_dep_time = st.sidebar.number_input("CRS Departure Time", value=df['CRSDepTime'].median())
crs_elapsed_time = st.sidebar.number_input("CRS Elapsed Time", value=df['CRSElapsedTime'].median())
year = st.sidebar.number_input("Year", value=int(df['Year'].median()))
quarter = st.sidebar.number_input("Quarter", value=int(df['Quarter'].median()))
month = st.sidebar.number_input("Month", value=int(df['Month'].median()))
day_of_month = st.sidebar.number_input("Day of Month", value=int(df['DayofMonth'].median()))
day_of_week = st.sidebar.number_input("Day of Week", value=int(df['DayOfWeek'].median()))
marketing_airline_network = st.sidebar.selectbox("Marketing Airline Network", df["Marketing_Airline_Network"].unique())
operated_or_branded_code_share_partners = st.sidebar.selectbox("Operated or Branded Code Share Partners", df["Operated_or_Branded_Code_Share_Partners"].unique())
iata_code_marketing_airline = st.sidebar.selectbox("IATA Code Marketing Airline", df["IATA_Code_Marketing_Airline"].unique())
operating_airline = st.sidebar.selectbox("Operating Airline", df["Operating_Airline"].unique())
iata_code_operating_airline = st.sidebar.selectbox("IATA Code Operating Airline", df["IATA_Code_Operating_Airline"].unique())


# Creating the predict button
if st.button("Predict"):

    # Creating the dataframe for the input values
    pred_row = spark.createDataFrame([[distance, year, quarter, month, day_of_month, day_of_week, crs_arr_time, crs_dep_time, crs_elapsed_time, airline, origin, dest, marketing_airline_network, operated_or_branded_code_share_partners, iata_code_marketing_airline, operating_airline, iata_code_operating_airline]],
                                     
                                    ['Distance', 'Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'CRSArrTime', 'CRSDepTime', 'CRSElapsedTime', 'Airline', 'Origin', 'Dest', 'Marketing_Airline_Network', 'Operated_or_Branded_Code_Share_Partners', 'IATA_Code_Marketing_Airline', 'Operating_Airline', 'IATA_Code_Operating_Airline'])


    # Transforming the input values from the string indexer
    pred_row = indexer.transform(pred_row)


    # Selecting only the columns that are needed for the model
    col_path = 'cols.txt'
    file_contents = []

    with open(col_path, "r") as file:
        for line in file:
            file_contents.append(line.strip())

    pred_row = pred_row.select(file_contents)


    # Predicting the outcome using the model
    pred_row = model.transform(pred_row)

    # Some formatting for the output
    pred_row = pred_row.select('prediction')
    pred = pred_row.collect()[0][0]
    outcome = "Your flight WILL be delayed" if pred > 0 else "Your flight WILL NOT be delayed"
    
    # Displaying the output with the Prediction
    st.write(outcome)