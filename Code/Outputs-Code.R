# Title: DC Residential Properties Outputs
# Author: Alexander Zakrzeski
# Date: December 27, 2024

# Part 1: Setup and Configuration

# Load to import, clean, and wrangle data
library(dplyr)
library(readr)
library(stringr)
library(tibble)
library(tidyr)

# Load to produce charts and tables
library(ggcorrplot)
library(ggplot2)
library(gt)
library(scales)

# Part 2: Data Preprocessing

# Load the data from the CSV file 
appraisals <- read_csv("Outputs-Data.csv")

# Select columns and reshape the data into a longer format
dens_inputs <- appraisals |> 
  select(price, log_price) |>
  pivot_longer(cols = everything(),
               names_to = "variable", 
               values_to = "value") |> 
  # Modify values in the column and sort the rows 
  mutate(variable = if_else(  
    variable == "price", "Price", "Log of Price"    
    ) |>            
                    factor(levels = c("Price", "Log of Price"))) |>  
  arrange(variable)

# Select columns, modify the values of columns, and rename columns
sr_corr_inputs <- appraisals |> 
  select(log_gba, ttl_bathrms, fireplaces, rooms, bedrms, age, saledate_y, 
         landarea, log_price) |> 
  mutate(across(c(ttl_bathrms, fireplaces, bedrms, saledate_y), 
                ~ as.numeric(factor(.x))),
         rooms = case_when(  
           rooms == "5 or Fewer" ~ 1, 
           rooms == "6" ~ 2, 
           rooms == "7" ~ 3,
           rooms == "8" ~ 4, 
           rooms == "9" ~ 5,
           rooms == "10 or More" ~ 6  
           ), 
         age = case_when(  
           age == "10 or Fewer" ~ 1,  
           age == "11 to 60" ~ 2, 
           age == "61 to 70" ~ 3,
           age == "71 to 80" ~ 4, 
           age == "81 to 90" ~ 5,
           age == "91 to 100" ~ 6, 
           age == "101 to 110" ~ 7,  
           age == "111 or More" ~ 8  
           ), 
         landarea = case_when( 
           landarea == "999 or Fewer" ~ 1,
           landarea == "1,000 to 1,999" ~ 2,
           landarea == "2,000 to 4,999" ~ 3, 
           landarea == "5,000 or More" ~ 4 
           )) |>
  rename(`Log of GBA` = log_gba, 
         Bathrooms = ttl_bathrms,
         Fireplaces = fireplaces,
         Rooms = rooms, 
         Bedrooms = bedrms,
         Age = age, 
         Year = saledate_y,
         `Land Area` = landarea,
         `Log of Price` = log_price)  

# Select columns and modify values in a column 
box_plot_inputs <- appraisals |>  
  select(ward, heat_d, extwall_d, roof_d, log_price) |>
  mutate(ward = as.character(ward)) |> 
  # Reshape the data into a longer format and modify values in a column
  pivot_longer(cols = -log_price, 
               names_to = "variable",
               values_to = "value") |> 
  mutate(variable = case_when(  
    variable == "ward" ~ "Ward", 
    variable == "heat_d" ~ "Heating System",  
    variable == "extwall_d" ~ "Exterior Wall",
    variable == "roof_d" ~ "Roof" 
    ) |>            
                    factor(c("Ward", "Heating System", "Exterior Wall", 
                             "Roof"))) |>   
  # Reorder a column and sort the rows
  relocate(log_price, .after = last_col()) |> 
  arrange(variable) 

# Create a tibble containing the necessary rows and columns
model_metrics <- tibble( 
  Model = c("Linear Regression", "k-Nearest Neighbors", 
            "Support Vector Machine", "Random Forest", "CatBoost", "XGBoost"),
  `R²` = c("0.828", "0.806", "0.865", "0.860", "0.873", "0.868"),
  RMSE = c("$210,608", "$223,941", "$190,687", "$190,045", "$182,136", 
           "$184,864"), 
  MAE = c("$153,450", "$158,017", "$134,425", "$132,210", "$129,190", 
          "$130,634")  
  )

# Part 3: Outputs

# Create a faceted density plot to display distributions
ggplot(dens_inputs, aes(x = value)) + 
  geom_density(fill = "#0078ae") +
  geom_hline(yintercept = 0, linewidth = 1.15, color = "#000000") + 
  scale_x_continuous(breaks = pretty_breaks(3), labels = comma) + 
  scale_y_continuous(breaks = pretty_breaks(4), 
                     labels = label_number(drop0trailing = TRUE)) + 
  labs(title = "Figure 1: Distributions of Price and Log of Price", 
       x = "", y = "Density") + 
  facet_wrap(~ variable, scales = "free") + 
  theme_void() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"), 
        panel.spacing.x = unit(2.25, "lines"),  
        panel.spacing.y = unit(1.25, "lines"),  
        text = element_text(family = "Roboto"),
        plot.title = element_text(margin = margin(0, 0, 15, 0), hjust = 0.5, 
                                  size = 17, face = "bold"), 
        strip.text = element_text(margin = margin(0, 0, 10, 0), size = 15),
        panel.grid.major = element_line(linetype = 3, linewidth = 0.3, 
                                        color = "#808080"), 
        axis.title.y = element_text(margin = margin(0, 10, 0, 0), angle = 90, 
                                    size = 15),
        axis.text.x = element_text(margin = margin(-2.5, 0, 0, 0), size = 13, 
                                   color = "#000000"), 
        axis.text.y = element_text(margin = margin(0, 5, 0, 0), size = 13, 
                                   color = "#000000")) 

# Create a correlation matrix to display the correlation coefficients
ggcorrplot(cor(sr_corr_inputs, method = "spearman"), legend.title = "",   
           outline.color = "#000000", lab = TRUE, lab_size = 3.5, 
           tl.cex = 11.25, tl.srt = 45) + 
  scale_fill_gradient2(low = "#c41230", mid = "#edeeee", high = "#5e9732", 
                       limits = c(-1, 1), 
                       labels = label_number(drop0trailing = TRUE)) + 
  labs(title = "Figure 2: Spearman's Rank Correlation Matrix",
       x = "", y = "") +
  guides(fill = guide_colorbar(title = "", ticks.colour = "#444547", 
                               frame.colour = "#444547", barwidth = 1.25, 
                               barheight = 21, 
                               label.theme = element_text(color = "#000000", 
                                                          size = 11.5))) + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"),
        text = element_text(family = "Roboto"), 
        plot.title = element_text(margin = margin(0, 0, 15, 0), hjust = 0.5, 
                                  size = 17, face = "bold"), 
        panel.grid.major = element_blank(), 
        axis.text = element_text(color = "#000000"))     

# Create a scatter plot to display the relationship between the variables
ggplot(appraisals, aes(x = log_gba, y = log_price)) +
  geom_point(color = "#0078ae", size = 2.25) +  
  geom_smooth(method = "lm", se = FALSE, linewidth = 1.15, color = "#000000") +
  geom_hline(yintercept = 12, linewidth = 1.15, color = "#000000") + 
  scale_x_continuous(breaks = pretty_breaks(4), 
                     labels = label_number(drop0trailing = TRUE)) + 
  scale_y_continuous(limits = c(12, 15.1), breaks = seq(12, 15.1, by = 1)) +
  labs(title = str_squish("Figure 3: Relationship between Log of Price and Log 
                           of Gross Building Area"), 
       x = "Log of Gross Building Area", y = "Log of Price") + 
  theme_void() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"),  
        text = element_text(family = "Roboto"),
        plot.title = element_text(margin = margin(0, 0, 15, 0), hjust = 0.5, 
                                  size = 17, face = "bold"), 
        panel.grid.major = element_line(linetype = 3, linewidth = 0.3, 
                                        color = "#808080"),
        axis.title.x = element_text(margin = margin(10, 0, 0, 0), size = 15),
        axis.title.y = element_text(margin = margin(0, 10, 0, 0), angle = 90, 
                                    size = 15),
        axis.text.x = element_text(margin = margin(-2.5, 0, 0, 0), size = 14, 
                                   color = "#000000"), 
        axis.text.y = element_text(margin = margin(0, 5, 0, 0), size = 14, 
                                   color = "#000000"))   

# Create a faceted box plot to display distributions
ggplot(box_plot_inputs, aes(x = value, y = log_price)) + 
  geom_boxplot(fill = "#3d98c1") + 
  geom_hline(yintercept = 12, linewidth = 0.6, color = "#000000") +
  scale_y_continuous(limits = c(12, 15), breaks = seq(12, 15, by = 1)) +
  labs(title = str_squish("Figure 4: Distributions of the Log of Price for the 
                           Nominal Variables"),  
       x = "", y = "") +
  facet_wrap(~ variable, scales = "free") +
  theme_void() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"),
        panel.spacing.x = unit(3.25, "lines"),
        panel.spacing.y = unit(2, "lines"), 
        text = element_text(family = "Roboto"), 
        plot.title = element_text(margin = margin(0, 0, 15, 0), hjust = 0.5, 
                                  size = 19, face = "bold"), 
        strip.text = element_text(margin = margin(0, 0, 10, 0), size = 15), 
        panel.grid.major.y = element_line(linetype = 3, linewidth = 0.3, 
                                          color = "#808080"), 
        axis.text.x = element_text(margin = margin(-2.5, 0, 0, 0), size = 13, 
                                   color = "#000000"), 
        axis.text.y = element_text(margin = margin(0, 5, 0, 0), size = 13, 
                                   color = "#000000")) 

# Create a table that displays the metrics of the models 
gt(model_metrics) |>
  tab_header(title = md("**Table 1: Results for Predicting D.C. Residential 
                           Property Sale Prices**")) |>
  cols_align(align = "center", columns = -c(Model)) |>
  tab_options(table.width = "90%", table.font.names = "Roboto", 
              table.font.size = px(18), data_row.padding = px(12.5)) |>
  opt_stylize(style = 6)