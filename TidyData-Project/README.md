# Tidy Data Project

## Overview
In this "tidy data" mini-project, I showcase my skills in data cleaning and visualization in Python, using a "messy" dataset on the medalists at the 2008 Summer Olympics in Beijing. With the help of methods like ``pd.melt()``, ``dropna()``, and ``str.split()``, I first transform the data according to the principles of "tidy data" put forth by New Zealand statistician Hadley Wickham. He proposed that data be formatted such that:

(1) Every variable has its own column.

(2) Every observation forms a separate row.

(3) Every distinct type of observational unit has its own separate table

Following this universal standard for data formatting makes cleaning, modeling, and visualization much simpler, since many of the tools we use for facilitating data analysis in Python (such as those found in the [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf) expect data to be inputted in this format. It also cuts down the number of tools necessary to transform messy data into the tidy data we want. For more information on "tidy data", check out Wickham's original article linked [here](https://www.jstatsoft.org/article/view/v059i10).

After cleaning the data, I end up with a pandas DataFrame containing information on 1,875 athletes at the 2008 Olympics, namely their ``Gender``, the ``Sport`` they played, and the type of ``Medal`` they earned (i.e. gold, silver, or bronze). I then explore the relationships between these three categorical variables through pivot tables and visualizations, and I discuss some possible reasons for the patterns we discover. 

Thank you for checking out my project, and I hope you learn a thing or two about data cleaning and visualization in Python!

## Getting Started
To run the Jupyter notebook yourself, you will first need to download the "TidyData-Project" folder from my data science portfolio repository. To do that, first go to [this link](https://download-directory.github.io/). This will open up a page that looks like this:

![Getting Started 1](data/Getting_Started_1.png)

Paste the following link into the box in the center of the page and press enter on the keyboard: https://github.com/t-clark04/Clark-Data-Science-Portfolio/tree/main/TidyData-Project.

The project files have now been downloaded to your computer as a zip file. Locate the zip file in your Downloads folder. It should look something like this:

![Getting Started 2](data/Getting_Started_2.png)
