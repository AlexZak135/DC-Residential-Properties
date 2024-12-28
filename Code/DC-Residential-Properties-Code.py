# Title: DC Residential Properties Analysis
# Author: Alexander Zakrzeski
# Date: December 27, 2024

# Part 1: Setup and Configuration

# Load to import, clean, and wrangle data  
import numpy as np
import pandas as pd

# Load to produce data visualizations
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotnine as pn
import seaborn as sns

# Load to run statistical tests
from scipy.stats import pearsonr, pointbiserialr
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load to train, test, and evaluate machine learning models
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error, root_mean_squared_error 
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Define a function for control flow using the conditions and choices
def conditional_map(*conditional_choice_pairs): 
    # Map conditions to corresponding choices 
    mapped_array = np.select( 
      condlist = conditional_choice_pairs[0::2], 
      choicelist = conditional_choice_pairs[1::2], 
      default = pd.NA  
      )  

    # Return the array
    return mapped_array

# Define a function to remove trailing ".0" from numeric strings
def remove_dot_zero(series):
    # Convert the series to strings and remove any trailing ".0"
    series = series.astype(str)
    series = series.str.replace(r"\.0$", "", regex = True) 
    
    # Return the series
    return series

# Define a function to plot a distribution using a density plot
def generate_density_plot(column, value1, value2): 
    # Create a density plot to display the distribution
    plot = (pn.ggplot(appraisals, pn.aes(x = column)) + 
      pn.geom_density(fill = "#0078ae") + 
      pn.scale_x_continuous(labels = lambda x: 
                                     ["{:,.0f}".format(i) for i in x]) + 
      pn.labs(title = f"{value1} Distribution of {value2}", 
              x = "", y = "Density") +
      pn.theme_bw() +
      pn.theme(plot_title = pn.element_text(hjust = 0.5), 
               text = pn.element_text(size = 14))) 
        
    # Display the plot
    plot.show()
         
# Define a function to plot distributions using a box plot
def generate_box_plot(column, value1, value2):
    # Create a box plot to display the distributions
    plot = (pn.ggplot(appraisals, pn.aes(x = column, y = "log_price")) + 
      pn.geom_boxplot(fill = "#3d98c1") + 
      pn.scale_y_continuous(labels = lambda x: [f"{i:g}" for i in x]) +
      pn.labs(title = f"{value1} Distribution of the Log of Price by {value2}", 
              x = "", y = "") +
      pn.theme_bw() +   
      pn.theme(plot_title = pn.element_text(hjust = 0.5), 
               text = pn.element_text(size = 14))) 
    
    # Display the plot
    plot.show()    

# Define a function to prepare the training and test data 
def process_and_split(model): 
    # Drop the columns that will no longer be used
    processed = (appraisals. 
      drop(columns = ["saledate", "heat_d", "num_units", "bedrms", "stories", 
                      "extwall_d", "floor_d", "kitchens", "price"])) 
    
    # Appropriately create dummy variables from the categorical variables
    if model == "lr": 
        processed = pd.get_dummies(processed, drop_first = True)
    elif model in ["knn", "svm", "rf", "xgb"]:  
        processed = pd.get_dummies(processed, drop_first = False) 
    
    # Modify the values of the column to display integers instead of booleans
    if model != "cb":
        processed[processed.select_dtypes(include = "bool").columns] = ( 
          processed.select_dtypes(include = "bool").astype(int) 
          )  
        
    # Perform a train-test split and reset the index of certain dataframes
    x_train, x_test, y_train, y_test = train_test_split( 
      processed.drop(columns = "log_price"), processed["log_price"], 
      test_size = 0.2, shuffle = False 
      )
    x_test, y_test = [ 
      dataframe.reset_index(drop = True) for dataframe in [x_test, y_test] 
      ]

    # Return the dataframes
    return x_train, x_test, y_train, y_test

# Part 2: Data Preprocessing

# Load the data from the CSV file 
appraisals = pd.read_csv("DC-Residential-Properties-Data.csv")

# Make all column names lowercase, rename specific ones, and drop columns
appraisals = (appraisals.
  rename(columns = str.lower).
  rename(columns = {"bathrm": "bathrms", 
                    "hf_bathrm": "hf_bathrms",
                    "bedrm": "bedrms", 
                    "intwall_d": "floor_d"}). 
  drop(columns = ["heat", "eyb", "style", "struct", "grade", "grade_d", 
                  "cndtn", "cndtn_d", "extwall", "roof", "intwall", 
                  "gis_last_mod_dttm", "objectid"])) 

# Filter based on multiple conditions to create a subsetted dataframe
appraisals = appraisals[ 
  (appraisals["bathrms"].between(1, 4)) & 
  (appraisals["hf_bathrms"].between(0, 2)) & 
 ~((appraisals["heat_d"].isin(["Air Exchng", "Evp Cool", "Ind Unit", 
                               "No Data"])) | 
   ((appraisals["heat_d"] == "Warm Cool") & (appraisals["ac"] == "N"))) & 
  (appraisals["ac"] != "0") & 
  (appraisals["num_units"].between(1, 4)) &
 ~((appraisals["num_units"] == 1) & (appraisals["struct_d"] == "Multi")) & 
  (appraisals["rooms"].between(1, 16)) & 
  (appraisals["rooms"] > appraisals["bedrms"]) & 
  (appraisals["bedrms"].between(1, 8)) &
  (appraisals["ayb"].between(1900, 2024)) & 
  (appraisals["ayb"] <= pd.to_datetime(appraisals["saledate"]).dt.year) & 
  (((appraisals["yr_rmdl"].between(appraisals["ayb"], 2024)) & 
    (appraisals["yr_rmdl"] <= pd.to_datetime(appraisals["saledate"]). 
                              dt.year)) | 
   (appraisals["yr_rmdl"].isna())) & 
  (((appraisals["stories"] == 1) & (appraisals["style_d"] == "1 Story")) |
   ((appraisals["stories"] == 1.5) & 
    (appraisals["style_d"] == "1.5 Story Fin")) | 
   ((appraisals["stories"] == 2) & (appraisals["style_d"] == "2 Story")) | 
   ((appraisals["stories"] == 2.5) & 
    (appraisals["style_d"] == "2.5 Story Fin")) |
   ((appraisals["stories"] == 3) & (appraisals["style_d"] == "3 Story")) | 
   ((appraisals["stories"] == 3.5) & 
    (appraisals["style_d"] == "3.5 Story Fin")) |
   ((appraisals["stories"] == 4) & (appraisals["style_d"] == "4 Story"))) &
  (pd.to_datetime(appraisals["saledate"]).dt.normalize() 
   .between(pd.Timestamp(2019, 9, 24, tz = "UTC"), 
            pd.Timestamp(2024, 9, 24, tz = "UTC"))) & 
  (appraisals["price"].between(250000, 2500000)) & 
  (appraisals["qualified"] == "Q") &
  (appraisals["sale_num"] <= 8) & 
  (appraisals["gba"].between(500, 5000)) &
  (appraisals["bldg_num"] == 1) & 
 ~((appraisals["struct_d"].isin(["No Data", "Vacant Land"])) | 
   ((appraisals["struct_d"] == "Multi") & 
    (appraisals["usecode"].isin([11, 13]))) | 
   ((appraisals["struct_d"].isin(["Row End", "Row Inside", "Town End", 
                                  "Town Inside"])) & 
    (appraisals["usecode"] == 12)) |
   ((appraisals["struct_d"] == "Semi-Detached") &
    (appraisals["usecode"] == 11)) |
   ((appraisals["struct_d"] == "Single") & 
    (appraisals["usecode"].isin([11, 13, 23, 24])))) &
 ~(appraisals["extwall_d"].isin(["Adobe", "Aluminum", "Concrete", 
                                 "Concrete Block", "Hardboard", "Metal Siding", 
                                 "No Data", "Plywood", "Rustic Log", 
                                 "SPlaster", "Stucco Block"])) &
 ~(appraisals["roof_d"].isin(["Concrete", "Concrete Tile", "Neopren", 
                              "Typical", "Water Proof", "Wood- FS"])) &
 ~(appraisals["floor_d"].isin(["Lt Concrete", "No Data", "Parquet", 
                               "Resiliant"])) &
  (appraisals["kitchens"].between(1, 4)) &
  (appraisals["fireplaces"] <= 3) &
  (appraisals["usecode"].isin([11, 12, 13, 23, 24])) &
  (appraisals["landarea"].between(500, 12000)) 
  ]

# Modify the values of existing columns and create new columns
appraisals["ssl"] = appraisals["ssl"].str.replace(r"\s{2,}", " ", regex = True) 
appraisals["ttl_bathrms"] = conditional_map(  
  appraisals["bathrms"] + (appraisals["hf_bathrms"] * 0.5) <= 3.5,
  remove_dot_zero(appraisals["bathrms"] + (appraisals["hf_bathrms"] * 0.5)),
  True, "4 or More"  
  ) 
appraisals["heat_d"] = conditional_map(  
  appraisals["heat_d"] == "Forced Air", "Forced Air",
  appraisals["heat_d"] == "Hot Water Rad", "Hot Water",
  appraisals["heat_d"] == "Warm Cool", "Dual Climate", 
  True, "Other"
  )
appraisals["ac"] = appraisals["ac"].replace({"Y": "Yes", "N": "No"})
appraisals["num_units"] = conditional_map( 
  appraisals["num_units"] == 1, "1", 
  True, "2 or More"
  )
appraisals["rooms"] = conditional_map( 
  appraisals["rooms"] <= 5, "5 or Fewer", 
 (appraisals["rooms"] >= 6) & (appraisals["rooms"] <= 9), 
  remove_dot_zero(appraisals["rooms"]), 
  appraisals["rooms"] >= 10, "10 or More"  
  ) 
appraisals["bedrms"] = conditional_map( 
  appraisals["bedrms"] <= 2, "2 or Fewer", 
 (appraisals["bedrms"] >= 3) & (appraisals["bedrms"] <= 4), 
  remove_dot_zero(appraisals["bedrms"]),
  appraisals["bedrms"] >= 5, "5 or More"
  )
appraisals["age"] = conditional_map( 
  pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] <= 10, 
  "10 or Fewer", 
((pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] >= 11) &
 (pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] <= 60)),
  "11 to 60", 
((pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] >= 61) &
 (pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] <= 70)), 
  "61 to 70",
((pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] >= 71) & 
 (pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] <= 80)), 
  "71 to 80",  
((pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] >= 81) &
 (pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] <= 90)), 
  "81 to 90",
((pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] >= 91) &
 (pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] <= 100)), 
  "91 to 100",
((pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] >= 101) &
 (pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] <= 110)), 
  "101 to 110", 
  pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] >= 111, 
  "111 or More"  
  ) 
appraisals["rmdl"] = conditional_map( 
 ~appraisals["yr_rmdl"].isna(), "Yes",
  True, "No" 
  )
appraisals["stories"] = conditional_map( 
  appraisals["stories"] <= 2, "2 or Fewer",
  True, "2.5 or More"
  )
appraisals["saledate"] = pd.to_datetime(appraisals["saledate"]).dt.date
appraisals["saledate_ym"] = (pd.to_datetime(appraisals["saledate"]). 
                             dt.to_period("M").dt.start_time.dt.date) 
appraisals["saledate_y"] = conditional_map( 
  pd.to_datetime(appraisals["saledate"]).dt.year.isin([2023, 2024]), "2023+", 
  True, pd.to_datetime(appraisals["saledate"]).dt.year.astype(str) 
  )
appraisals["log_price"] = np.log(appraisals["price"])
appraisals["log_gba"] = np.log(appraisals["gba"]) 
appraisals["extwall_d"] = conditional_map( 
  appraisals["extwall_d"].isin(["Brick Veneer", "Brick/Siding", "Brick/Stone", 
                                "Brick/Stucco", "Common Brick", 
                                "Face Brick"]), "Brick", 
  appraisals["extwall_d"].isin(["Shingle", "Stucco", "Vinyl Siding", 
                                "Wood Siding"]), "Siding and Stucco", 
  appraisals["extwall_d"].isin(["Stone", "Stone Veneer", "Stone/Siding", 
                                "Stone/Stucco"]), "Stone" 
  )
appraisals["roof_d"] = conditional_map( 
  appraisals["roof_d"].isin(["Clay Tile", "Slate"]), "Tile",
  appraisals["roof_d"].isin(["Comp Shingle", "Composition Ro", "Shake", 
                             "Shingle"]), "Shingle", 
  appraisals["roof_d"].isin(["Metal- Cpr", "Metal- Pre", 
                             "Metal- Sms"]), "Metal",
  appraisals["roof_d"] == "Built Up", "Flat"  
  )
appraisals["floor_d"] = conditional_map( 
  appraisals["floor_d"].isin(["Carpet", "Vinyl Sheet"]), "Soft", 
  True, "Hard"
  )
appraisals["kitchens"] = conditional_map( 
  appraisals["kitchens"] == 1, "1",   
  True, "2 or More" 
  )
appraisals["fireplaces"] = conditional_map( 
  appraisals["fireplaces"] <= 1, remove_dot_zero(appraisals["fireplaces"]),  
  True, "2 or More" 
  )
appraisals["landarea"] = conditional_map( 
  appraisals["landarea"] <= 999, "999 or Fewer",  
 (appraisals["landarea"] >= 1000) & (appraisals["landarea"] <= 1999),
  "1,000 to 1,999",
 (appraisals["landarea"] >= 2000) & (appraisals["landarea"] <= 4999), 
  "2,000 to 4,999", 
  appraisals["landarea"] >= 5000, "5,000 or More"
  )

# Load the data from the CSV file
properties = pd.read_csv("Property-Zoning-Data.csv", usecols = ["SSL", "Ward"])

# Make all column names lowercase and drop rows with missing values
properties = properties.rename(columns = str.lower).dropna() 

# Filter based on the condition and modify the values of existing columns 
properties = properties[~properties["ssl"].str.contains(r"^0000")]
properties["ssl"] = properties["ssl"].str.replace(r"\s{2,}", " ", regex = True)
properties["ward"] = properties["ward"].str.replace(r"^Ward ", "", 
                                                    regex = True) 

# Perform a left join, drop rows with missing values, and drop columns
appraisals = (appraisals.
  merge(properties, on = "ssl", how = "left").
  dropna(subset = "ward").
  drop(columns = ["ssl", "bathrms", "hf_bathrms", "ayb", "yr_rmdl", 
                  "qualified", "sale_num", "gba", "bldg_num", "style_d", 
                  "struct_d", "usecode", "saledate_ym"])) 

# Reorder the columns, sort the rows in ascending order, and reset the index
appraisals.insert(0, "saledate", appraisals.pop("saledate"))
appraisals.insert(1, "saledate_y", appraisals.pop("saledate_y"))
appraisals.insert(2, "ward", appraisals.pop("ward"))
appraisals.insert(3, "age", appraisals.pop("age"))
appraisals.insert(4, "rmdl", appraisals.pop("rmdl"))
appraisals.insert(5, "ttl_bathrms", appraisals.pop("ttl_bathrms"))
appraisals.insert(19, "price", appraisals.pop("price"))
appraisals.insert(20, "log_price", appraisals.pop("log_price"))
appraisals.insert(12, "log_gba", appraisals.pop("log_gba"))
appraisals = appraisals.sort_values(by = "saledate").reset_index(drop = True)

# Part 3: Exploratory Data Analysis

# Generate summary statistics
print(appraisals.select_dtypes("number").describe().map(lambda x: f"{x:.2f}"))
     
# Create density plots to display distributions
generate_density_plot("price", "Figure 1:", "Price")
generate_density_plot("log_price", "Figure 2:", "the Log of Price")

# Select columns and modify the values of existing columns
sr_corr_inputs = (appraisals 
  [["log_price", "log_gba", "ttl_bathrms", "fireplaces", "rooms", "bedrms", 
    "age", "saledate_y", "landarea"]].copy())
sr_corr_inputs["ttl_bathrms"] = conditional_map( 
  sr_corr_inputs["ttl_bathrms"] == "1", 1,
  sr_corr_inputs["ttl_bathrms"] == "1.5", 2, 
  sr_corr_inputs["ttl_bathrms"] == "2", 3,
  sr_corr_inputs["ttl_bathrms"] == "2.5", 4,
  sr_corr_inputs["ttl_bathrms"] == "3", 5,
  sr_corr_inputs["ttl_bathrms"] == "3.5", 6,
  sr_corr_inputs["ttl_bathrms"] == "4 or More", 7   
  )
sr_corr_inputs["fireplaces"] = conditional_map( 
  sr_corr_inputs["fireplaces"] == "0", 1,
  sr_corr_inputs["fireplaces"] == "1", 2,
  sr_corr_inputs["fireplaces"] == "2 or More", 3
  )
sr_corr_inputs["bedrms"] = conditional_map(  
  sr_corr_inputs["bedrms"] == "2 or Fewer", 1,
  sr_corr_inputs["bedrms"] == "3", 2, 
  sr_corr_inputs["bedrms"] == "4", 3,
  sr_corr_inputs["bedrms"] == "5 or More", 4
  )
sr_corr_inputs["saledate_y"] = conditional_map(  
  sr_corr_inputs["saledate_y"] == "2019", 1,
  sr_corr_inputs["saledate_y"] == "2020", 2,
  sr_corr_inputs["saledate_y"] == "2021", 3,
  sr_corr_inputs["saledate_y"] == "2022", 4,
  sr_corr_inputs["saledate_y"] == "2023+", 5
  )
sr_corr_inputs["rooms"] = conditional_map( 
  sr_corr_inputs["rooms"] == "5 or Fewer", 1,
  sr_corr_inputs["rooms"] == "6", 2,
  sr_corr_inputs["rooms"] == "7", 3,
  sr_corr_inputs["rooms"] == "8", 4,
  sr_corr_inputs["rooms"] == "9", 5, 
  sr_corr_inputs["rooms"] == "10 or More", 6
  )
sr_corr_inputs["age"] = conditional_map(
  sr_corr_inputs["age"] == "10 or Fewer", 1,
  sr_corr_inputs["age"] == "11 to 60", 2, 
  sr_corr_inputs["age"] == "61 to 70", 3,
  sr_corr_inputs["age"] == "71 to 80", 4,
  sr_corr_inputs["age"] == "81 to 90", 5,
  sr_corr_inputs["age"] == "91 to 100", 6,
  sr_corr_inputs["age"] == "101 to 110", 7,
  sr_corr_inputs["age"] == "111 or More", 8 
  )
sr_corr_inputs["landarea"] = conditional_map( 
  sr_corr_inputs["landarea"] == "999 or Fewer", 1,
  sr_corr_inputs["landarea"] == "1,000 to 1,999", 2,
  sr_corr_inputs["landarea"] == "2,000 to 4,999", 3,
  sr_corr_inputs["landarea"] == "5,000 or More", 4
  )

# Rename columns and calculate Spearman's rank correlation coefficients 
sr_corr_inputs = (sr_corr_inputs. 
  rename(columns = {"log_price": "Log of Price",
                    "log_gba": "Log of GBA", 
                    "ttl_bathrms": "Bathrooms", 
                    "fireplaces": "Fireplaces", 
                    "rooms": "Rooms",
                    "bedrms": "Bedrooms",
                    "age": "Age",
                    "saledate_y": "Year",
                    "landarea": "Land Area"}).
  corr(method = "spearman").round(2))

# Create a correlation matrix to display the correlation coefficients 
plt.figure(figsize = (8, 6))
corr_matrix = sns.heatmap( 
  sr_corr_inputs, vmin = -1, vmax = 1, 
  cmap = sns.diverging_palette(h_neg = 10, h_pos = 240, as_cmap = True), 
  annot = True, annot_kws = {"color": "black"}, linewidths = 0.5, 
  linecolor = "black", 
  cbar_kws = {"format": ticker.FuncFormatter( 
    lambda x, _: f"{int(x)}" if x in [-1, 0, 1] else f"{x:.2f}" 
    )},
  xticklabels = True, yticklabels = True 
  )
corr_matrix.set_title(label = "Figure 3: Spearman's Rank Correlation Matrix", 
                      fontdict = {"fontsize": 18}, pad = 22) 
corr_matrix.tick_params(bottom = False, left = False)
plt.xticks(rotation = 45, ha = "right")
plt.show()

# Select columns and modify values in columns
pb_corr_inputs = (appraisals 
  [["rmdl", "ac", "num_units", "kitchens", "stories", "floor_d", 
    "log_price"]].copy())
pb_corr_inputs[["rmdl", "ac"]] = ( 
  (pb_corr_inputs[["rmdl", "ac"]] == "Yes").astype(int) 
  )
pb_corr_inputs[["num_units", "kitchens"]] = ( 
  (pb_corr_inputs[["num_units", "kitchens"]] == "2 or More").astype(int) 
  )
pb_corr_inputs["stories"] = ( 
  (pb_corr_inputs["stories"] == "2.5 or More").astype(int) 
  )
pb_corr_inputs["floor_d"] = (pb_corr_inputs["floor_d"] == "Hard").astype(int)  
         
# Generate Pearson and point-biserial correlation coefficients
print("\nThe correlation between log_gba and log_price:", 
      pearsonr(appraisals["log_gba"], 
               appraisals["log_price"]).statistic.round(2), "\n")
for column in ["rmdl", "ac", "num_units", "kitchens", "stories", "floor_d"]: 
    print(f"The correlation between {column} and log_price:", 
          pointbiserialr(pb_corr_inputs[column], 
                         pb_corr_inputs["log_price"]).statistic.round(2), "\n")

# Create a scatter plot to display the relationship between the variables
(pn.ggplot(appraisals, pn.aes(x = "log_gba", y = "log_price")) + 
 pn.geom_point(color = "#0078ae", size = 0.6) +
 pn.geom_smooth(method = "lm", se = False, size = 0.7) + 
 pn.scale_x_continuous(labels = lambda x: [f"{i:g}" for i in x]) +
 pn.scale_y_continuous(labels = lambda x: [f"{i:g}" for i in x]) +
 pn.labs(title = "Figure 4: Log of Price vs. Log of Gross Building Area", 
         x = "Log of Gross Building Area", y = "Log of Price") +
 pn.theme_bw() +
 pn.theme(plot_title = pn.element_text(hjust = 0.5), 
          text = pn.element_text(size = 14))).show() 

# Create box plots to display distributions
generate_box_plot("ward", "Figure 5:", "Ward")
generate_box_plot("heat_d", "Figure 6:", "Heating System")
generate_box_plot("extwall_d", "Figure 7:", "Exterior Wall")
generate_box_plot("roof_d", "Figure 8:", "Roof")

# Perform one-way ANOVAs and Tukey HSD post-hoc tests
for column in ["ward", "heat_d", "extwall_d", "roof_d"]:    
    print(f"\nOutputs for {column}:\n\n", 
          sm.stats.anova_lm(ols(f"log_price ~ {column}",    
                                data = appraisals).fit(), typ = 1), "\n\n", 
          pairwise_tukeyhsd(appraisals["log_price"], appraisals[column]), "\n")

# Part 4: Machine Learning Models

# Perform the train-test split for the model 
x1_train, x1_test, y1_train, y1_test = process_and_split("lr")

# Fit the model to the training data
lr_fit = LinearRegression().fit(x1_train, y1_train)

# Create a dataframe containing the performance and error metrics
model_metrics = pd.DataFrame({  
  "Model": "Linear Regression",
  "R\u00b2": format(lr_fit.score(x1_test, y1_test), ".3f"),
  "RMSE": "$" + format(root_mean_squared_error(    
    np.exp(y1_test), 
    np.exp(lr_fit.predict(x1_test)) *  
    np.mean(np.exp(y1_train - lr_fit.predict(x1_train))) 
    ), ",.0f"),
  "MAE": "$" + format(mean_absolute_error(  
    np.exp(y1_test),
    np.exp(lr_fit.predict(x1_test)) *  
    np.mean(np.exp(y1_train - lr_fit.predict(x1_train)))  
    ), ",.0f"),  
  }, index = [0]) 

# Perform the train-test split for the model, which will be reusable for others
x2_train, x2_test, y2_train, y2_test = process_and_split("knn")

# Tune hyperparameters with cross-validation to find the best hyperparameters
knn_best_hp = GridSearchCV(
  estimator = KNeighborsRegressor(),
  param_grid = {"n_neighbors": [15, 16, 17], 
                "weights": ["distance", "uniform"]}, 
  scoring = "neg_root_mean_squared_error",
  cv = KFold(n_splits = 5, shuffle = False) 
  ).fit(x2_train, y2_train).best_params_

# Fit the model to the training data
knn_fit = KNeighborsRegressor( 
  n_neighbors = knn_best_hp["n_neighbors"], 
  weights = knn_best_hp["weights"] 
  ).fit(x2_train, y2_train)

# Append performance and error metrics to an existing dataframe
model_metrics = pd.concat([model_metrics, pd.DataFrame({ 
  "Model": "k-Nearest Neighbors",
  "R\u00b2": format(knn_fit.score(x2_test, y2_test), ".3f"),
  "RMSE": "$" + format(root_mean_squared_error(   
    np.exp(y2_test), 
    np.exp(knn_fit.predict(x2_test)) * 
    np.mean(np.exp(y2_train - knn_fit.predict(x2_train)))  
    ), ",.0f"), 
  "MAE": "$" + format(mean_absolute_error(  
    np.exp(y2_test),
    np.exp(knn_fit.predict(x2_test)) *  
    np.mean(np.exp(y2_train - knn_fit.predict(x2_train)))  
    ), ",.0f"),    
  }, index = [0])], ignore_index = True) 

# Tune hyperparameters with cross-validation to find the best hyperparameters
svm_best_hp = GridSearchCV(   
  estimator = SVR(),
  param_grid = {"kernel": ["rbf"],
                "C": [28, 29, 30], 
                "epsilon": [0.05, 0.1, 0.2]},
  scoring = "neg_root_mean_squared_error",
  cv = KFold(n_splits = 5, shuffle = False)
  ).fit(x2_train, y2_train).best_params_

# Fit the model to the training data
svm_fit = SVR(
  kernel = svm_best_hp["kernel"],
  C = svm_best_hp["C"],
  epsilon = svm_best_hp["epsilon"] 
  ).fit(x2_train, y2_train)

# Append performance and error metrics to an existing dataframe
model_metrics = pd.concat([model_metrics, pd.DataFrame({  
  "Model": "Support Vector Machine",
  "R\u00b2": format(svm_fit.score(x2_test, y2_test), ".3f"),
  "RMSE": "$" + format(root_mean_squared_error(  
    np.exp(y2_test), 
    np.exp(svm_fit.predict(x2_test)) * 
    np.mean(np.exp(y2_train - svm_fit.predict(x2_train)))
    ), ",.0f"), 
  "MAE": "$" + format(mean_absolute_error( 
    np.exp(y2_test),
    np.exp(svm_fit.predict(x2_test)) *  
    np.mean(np.exp(y2_train - svm_fit.predict(x2_train)))  
    ), ",.0f"),  
  }, index = [0])], ignore_index = True)   

# Tune hyperparameters with cross-validation to find the best hyperparameters
rf_best_hp = GridSearchCV( 
  estimator = RandomForestRegressor(random_state = 123), 
  param_grid = {"n_estimators": [500],
                "max_depth": [17, 18, 19],
                "min_samples_split": [6, 7, 8], 
                "min_samples_leaf": [1]},
  scoring = "neg_root_mean_squared_error",
  cv = KFold(n_splits = 5, shuffle = False)  
  ).fit(x2_train, y2_train).best_params_

# Fit the model to the training data
rf_fit = RandomForestRegressor(  
  n_estimators = rf_best_hp["n_estimators"],
  max_depth = rf_best_hp["max_depth"],
  min_samples_split = rf_best_hp["min_samples_split"],
  min_samples_leaf = rf_best_hp["min_samples_leaf"], 
  random_state = 123
  ).fit(x2_train, y2_train)
  
# Append performance and error metrics to an existing dataframe
model_metrics = pd.concat([model_metrics, pd.DataFrame({ 
  "Model": "Random Forest",
  "R\u00b2": format(rf_fit.score(x2_test, y2_test), ".3f"),
  "RMSE": "$" + format(root_mean_squared_error(   
    np.exp(y2_test), 
    np.exp(rf_fit.predict(x2_test)) * 
    np.mean(np.exp(y2_train - rf_fit.predict(x2_train)))  
    ), ",.0f"), 
  "MAE": "$" + format(mean_absolute_error( 
    np.exp(y2_test),
    np.exp(rf_fit.predict(x2_test)) *  
    np.mean(np.exp(y2_train - rf_fit.predict(x2_train)))  
    ), ",.0f"),     
  }, index = [0])], ignore_index = True)   

# Perform the train-test split for the model and then create the pool objects
x3_train, x3_test, y3_train, y3_test = process_and_split("cb")
train_pool = Pool( 
  data = x3_train, label = y3_train, 
  cat_features = x3_train.drop(columns = "log_gba").columns.tolist() 
  )
test_pool = Pool( 
  data = x3_test, label = y3_test, 
  cat_features = x3_test.drop(columns = "log_gba").columns.tolist() 
  )

# Tune hyperparameters with cross-validation to find the best hyperparameters
cb_best_hp = pd.DataFrame(
  [CatBoostRegressor(loss_function = "RMSE", silent = True, 
                     random_state = 123). 
   grid_search(param_grid = {"iterations": [500],
                             "learning_rate": [0.05, 0.1, 0.2],
                             "depth": [6, 7, 8], 
                             "l2_leaf_reg": [2, 3, 4]}, 
               X = train_pool,
               cv = 5,
               shuffle = False, 
               verbose = False)["params"]],  
  index = [0])

# Fit the model to the training data
cb_fit = CatBoostRegressor(
  iterations = cb_best_hp["iterations"].iloc[0],
  learning_rate = cb_best_hp["learning_rate"].iloc[0],
  depth = cb_best_hp["depth"].iloc[0],
  l2_leaf_reg = cb_best_hp["l2_leaf_reg"].iloc[0], 
  silent = True,
  random_state = 123 
  ).fit(train_pool)

# Append performance and error metrics to an existing dataframe
model_metrics = pd.concat([model_metrics, pd.DataFrame({ 
  "Model": "CatBoost",
  "R\u00b2": format(cb_fit.score(test_pool), ".3f"),
  "RMSE": "$" + format(root_mean_squared_error( 
    np.exp(y3_test),
    np.exp(cb_fit.predict(test_pool)) *
    np.mean(np.exp(y3_train - cb_fit.predict(train_pool))) 
    ), ",.0f"),
  "MAE": "$" + format(mean_absolute_error( 
    np.exp(y3_test),
    np.exp(cb_fit.predict(test_pool)) *
    np.mean(np.exp(y3_train - cb_fit.predict(train_pool))) 
    ), ",.0f"),
  }, index = [0])], ignore_index = True)

# Tune hyperparameters with cross-validation to find the best hyperparameters
xgb_best_hp = GridSearchCV( 
  estimator = XGBRegressor(random_state = 123), 
  param_grid = {"n_estimators": [500],
                "max_depth": [5, 6, 7],
                "learning_rate": [0.025, 0.05, 0.1],
                "min_child_weight": [3, 4, 5], 
                "subsample": [1],
                "colsample_bytree": [0.3, 0.4, 0.5]}, 
  scoring = "neg_root_mean_squared_error",
  cv = KFold(n_splits = 5, shuffle = False)  
  ).fit(x2_train, y2_train).best_params_

# Fit the model to the training data
xgb_fit = XGBRegressor( 
  n_estimators = xgb_best_hp["n_estimators"],  
  max_depth = xgb_best_hp["max_depth"], 
  learning_rate = xgb_best_hp["learning_rate"], 
  min_child_weight = xgb_best_hp["min_child_weight"],
  subsample = xgb_best_hp["subsample"],
  colsample_bytree = xgb_best_hp["colsample_bytree"],
  random_state = 123
  ).fit(x2_train, y2_train) 

# Append performance and error metrics to an existing dataframe
model_metrics = pd.concat([model_metrics, pd.DataFrame({ 
  "Model": "XGBoost",
  "R\u00b2": format(xgb_fit.score(x2_test, y2_test), ".3f"),
  "RMSE": "$" + format(root_mean_squared_error(   
    np.exp(y2_test), 
    np.exp(xgb_fit.predict(x2_test)) * 
    np.mean(np.exp(y2_train - xgb_fit.predict(x2_train)))  
    ), ",.0f"),
  "MAE": "$" + format(mean_absolute_error( 
    np.exp(y2_test),
    np.exp(xgb_fit.predict(x2_test)) *  
    np.mean(np.exp(y2_train - xgb_fit.predict(x2_train)))  
    ), ",.0f"),   
  }, index = [0])], ignore_index = True)
