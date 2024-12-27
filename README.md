# Predicting Residential Property Sale Prices in Washington, D.C.

## Overview

The nation’s capital, Washington, D.C., is filled with a vast number of residential properties, many existing for well over a century, while others are brand-new. These residential properties display numerous architectural styles and reflect the city’s history as well as the incredible diversity that makes the district unique compared to any other place in the United States. Washington, D.C., consists of eight respective wards, with each ward varying in its residential properties, ranging from modest starter homes for new families to massive mansions inhabited by the district’s elite. Though differences exist in the styles and sizes of the properties, each property is special and unique to Washington, D.C., contributing to the incredible landscape of residential properties that define the district.

The government of Washington, D.C., is tasked with the responsibility of accurately predicting the appraisal prices of residential properties, as a portion of these predicted appraisal prices constitutes the ad valorem property tax that owners must pay to the district. To fulfill this duty, the Office of Tax and Revenue’s Computer-Assisted Mass Appraisal (CAMA) database, maintained by the Assessment Division within the Real Property Tax Administration, serves as the data source for developing techniques to predict appraisal prices, from which property taxes are derived. This database contains a comprehensive sale history of active properties listed in the district’s real property tax assessment roll. Furthermore, the database contains an extensive number of characteristics of these properties at the time of their sales, and the contents of the database are updated daily.

This research utilized data from the CAMA database, as well as property zoning data maintained by Washington, D.C. The data consisted of five years of residential property sales in Washington, D.C., spanning from September 24, 2019, to September 24, 2024. Data preprocessing was completed before any exploratory analysis, including modifying values in columns, addressing missing values, filtering based on specific logic, and creating new variables from existing ones for feature engineering purposes. In the exploratory data analysis, various statistical tests and methods were implemented, such as Spearman’s rank, Pearson, Point-Biserial Correlation tests, and ANOVA, along with the inspection of descriptive statistics to measure both central tendency and dispersion. The data was extensively visualized using density plots and box plots to examine distributions, as well as scatter plots to explore relationships between numeric variables.

After data preprocessing and exploratory data analysis, six supervised machine learning models were trained and tested, with three of these models using tree-based ensemble algorithms. All the models were trained on the initial 80% of the data and tested on the remaining 20%. Among the six models, the one using the CatBoost algorithm demonstrated the best performance based on the evaluated metrics, with XGBoost closely trailing. The machine learning models could be assessed by the Washington, D.C. government to determine if they have greater explanatory power and precision than current methods for predicting appraisal prices of residential properties, from which property taxes are derived. 

## Files

This is the [Python script](https://github.com/AlexZak135/DC-Residential-Properties/blob/main/Code/DC-Residential-Properties-Code.py) containing the code used for this analysis, and these are the [datasets](https://github.com/AlexZak135/DC-Residential-Properties/tree/main/Data) used in the Python script.

## Outputs

Below, Table 1 presents the performance metrics of all the models when tested on Washington, D.C. residential property sales from July 11, 2023, to September 24, 2024. 

<img width="1382" alt="Output-1" src="https://github.com/user-attachments/assets/a3d9ad90-a27c-4343-a131-b2e8ec836606" />
