# Predicting D.C. Residential Property Sale Prices

## Overview

The nation’s capital, Washington, D.C., is filled with a vast number of residential properties, many existing for well over a century, while others are brand-new. These residential properties display numerous architectural styles and reflect the city’s history as well as the incredible diversity that makes the district unique compared to any other place in the United States. Washington, D.C., consists of eight respective wards, with each ward varying in its residential properties, ranging from modest starter homes for new families to massive mansions inhabited by the district’s elite. Though differences exist in the styles and sizes of the properties, each property is special and unique to Washington, D.C., contributing to the incredible landscape of residential properties that define the district.

The government of Washington, D.C., is tasked with the responsibility of accurately predicting the appraisal prices of residential properties, as a portion of these predicted appraisal prices constitutes the ad valorem property tax that owners must pay to the district. To fulfill this duty, the Office of Tax and Revenue’s Computer-Assisted Mass Appraisal (CAMA) database, maintained by the Assessment Division within the Real Property Tax Administration, serves as the data source for developing techniques to predict appraisal prices, from which property taxes are derived. This database contains a comprehensive sale history of active properties listed in the district’s real property tax assessment roll. Furthermore, the database contains an extensive number of characteristics of these properties at the time of their sales, and the contents of the database are updated daily.

This research utilized data from the CAMA database and property zoning records maintained by Washington, D.C., spanning five years of residential property sales from September 24, 2019, to September 24, 2024. Following data preprocessing, exploratory analysis involved statistical methods such as Spearman’s rank, Pearson, point-biserial correlation, and ANOVA, paired with visualizations like density plots, box plots, and scatter plots. Six supervised machine learning models were developed, including three tree-based ensemble algorithms, with training conducted on the initial 80% of the data in ascending time order and testing on the remaining 20%. Of these models, CatBoost demonstrated the strongest performance, closely followed by XGBoost. The models have the potential to help the Washington, D.C. government achieve more accurate predictions for residential property appraisal prices, which serve as the basis for property taxes.

## Files

This is the [Python script](https://github.com/AlexZak135/DC-Residential-Properties/blob/main/Code/DC-Residential-Properties-Code.py) containing the code used for this analysis, and these are the [datasets](https://github.com/AlexZak135/DC-Residential-Properties/tree/main/Data) used in the Python script.

## Outputs

These [outputs](https://github.com/AlexZak135/DC-Residential-Properties/tree/main/Outputs) represent the main findings and results from the exploratory data analysis and the machine learning models.
