# Green Energy Cost and Production Predictor

A tool that predicts the cost and energy output of a wind farm or solar array
in the United States based on locational inputs like longitude and latitude.

## Introduction and Motivation

In the modern day, climate change is an issue that becomes more prevalent
with each passing day. With this, it is important to ensure that we, as
a society, minimize our contributions to accelerating climate change
unnaturally. One major way that this is done is by utilizing green
energy technology like wind turbines or solar panels. In the United States,
there are a wide range of biomes and ecosystems that each experience
varying conditions like wind speed and solar irradiance. This makes it
somewhat difficult to make use of green energy technologies just anywhere.
This tool seeks to alleviate this difficulty by allowing for the quick access
to accurate predictions of energy output and cost of a potential wind farm or
solar array of a certain size in a certain location in the United States.

## Running the Artifact

The notebooks, and their outputs are available to be examined in the
[src directory](/src/) at anytime, although the outputs could be outdated,
nonexistent, or other inconsistencies could arise, so it is best to run them
yourself. Below are some simple steps to get up and running with this project.

This artifact assumes [Python 3.11.X](https://www.python.org/downloads/)
and [Poetry](https://python-poetry.org/docs/) are installed prior to
any of the steps below.

To use this prototype, follow the steps below:

- Clone or fork the repository
- In the base of the repository, type `poetry shell` to create an isolated
    virtual environment
- Type `poetry install` to get all of the dependencies
- Type `jupyter notebook` to boot up a Jupyter server on your localhost

Now, you are free to explore the data by sleuthing through the directories and
by running the notebooks in this server. However, if a guided experience is
desired, it is suggested to start with `eda.ipynb` to get a grasp of what all
of this data means. From there, the notebooks will guide you through each
cell and through each notebook, giving you the all-inclusive green energy
prediction experience!

## Some Related Works

Some similar work has been done in the past. The largest one, NREL's PVWatts,
gives very detailed solar irradiance data based on location, and many other
factors pertaining specifically to solar panels. This, and other related tools,
work, and articles are listed below:

- NREL's PVWatts: [API Docs](https://developer.nrel.gov/docs/solar/pvwatts/v8/),
    [Online Calculator](https://pvwatts.nrel.gov/)
- NREL's WIND Toolkit Power Data: [Full Report](https://www.nrel.gov/docs/fy16osti/66189.pdf),
    [Data](https://data.nrel.gov/submissions/54)
- ML for Solar Energy Production: [GitHub](https://github.com/ColasGael/Machine-Learning-for-Solar-Energy-Prediction)
- Solarpy: [GitHub](https://github.com/aqreed/solarpy), [Inspiring Writing](https://www.eng.uc.edu/~beaucag/Classes/SolarPowerForAfrica/Solar%20Engineering%20of%20Thermal%20Processes,%20Photovoltaics%20and%20Wind.pdf)
- Windpowerlib: [GitHub](https://github.com/wind-python/windpowerlib),
    [Inspiring Work](https://github.com/oemof/feedinlib)
- Windtools: [GitHub](https://github.com/FZJ-IEK3-VSA/windtools)
- Open Sustainable Technology: [GitHub](https://github.com/protontypes/open-sustainable-technology)
- Climate Change 2021: [Report](https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_FrontMatter.pdf)
- Climate Change Impacts On Wind Power Generation: [Article](https://www.nature.com/articles/s43017-020-0101-7)
- Renewable Energy and Climate Change: [Article](https://www.sciencedirect.com/science/article/pii/S1364032122000405)
- Wind and Solar Effectiveness: [Article](https://www.sciencedirect.com/science/article/pii/S1342937X23000369)

## Technical Details

To perform these predictions, a sufficient swath of data had to be collected,
as machine learning techniques would be used to analyze data and make
predictions. This data can be found in the `data` directory, and consists
of two data sets: `wind.csv` and `solar.csv`. These two data sets
each consist of 9 attributes, with each attribute having either approximately
125000 or 11500 observations depending on the data set being examined. Full
diagnostic reports regarding these data sets can be found in the
[exploratory data analysis Jupyter Notebook](src/eda.ipynb).

These data sets are the heart of this artifact, in that they are what drives
the predictions to be made. Making use of Python Jupyter Notebooks, the data
are read in as Pandas data frames and passed into various Scikit-Learn
machine learning tools. These include linear regressions, k-means clustering,
and bisecting k-means clustering. These tools, along with Matplotlib,
allow for drawing connections between the data through graphical
representations.

## Future Plans

Moving forward, this artifact prototype would be expanded in various ways. One
way would be to gather more useful data if necessary. This could allow for more
analysis, and potentially for stronger correlations to be drawn, which could
prove this work even more useful. The largest avenue for expansion, however,
would include more in-depth analysis of clustering algorithms, or any other
useful machine learning models provided in the Scikit-Learn Python package.
The most important aspect of this would be measuring the accuracy of
predictions through cluster evaluation metrics, like Dunn's Index, as well as
cophenetic correlation coefficient. These will help to evaluate how the
clusters being created are performing in multiple aspects. The final piece
of future work laid out as of now, is to host this analysis on a website,
and give that website the ability to have tweakable attributes so users can
see the results in relative real-time.
