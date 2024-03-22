# Renewable ML

A [Quarto](https://quarto.org/) website that documents the computational ,
experimental, and analytical processes of training, testing, and evaluating
of machine learning algorithms.

## Introduction and Motivation

In the modern day, climate change is an issue that becomes more prevalent
with each passing day. With this, it is important to ensure that we, as
a society, minimize our contributions to accelerating climate change
unnaturally. One major way that this is done is by utilizing green
energy technology like wind turbines or solar panels. In the United States,
there are a wide range of biomes and ecosystems that each experience
varying conditions like wind speed and solar irradiance. This makes it
somewhat difficult to make use of green energy technologies just anywhere.
This work seeks to alleviate this difficulty by allowing for the quick access
to accurate predictions of energy output and cost of a potential wind farm or
solar array of a certain size in a certain location in the United States,
through the power of machine learning.

## Running the Artifact

### Accessing the Website

The website can be found at this [link](https://aidanneeson610artifact.netlify.app/)

### Running the Notebooks Manually

If it is desired to run the notebooks manually and/or interact with the
datasets, `Jupyter` makes this possible through the use of a few simple
commands.

Renewable ML assumes [Python 3.11.X](https://www.python.org/downloads/)
and [Poetry](https://python-poetry.org/docs/) are installed prior to
any of the steps below.

To run the notebooks, follow the steps below:

- Clone or fork the repository
- In the base of the repository, type `poetry shell` to create an isolated
    virtual environment
- Type `poetry install` to get all of the dependencies
- Type `jupyter notebook` to boot up a Jupyter server on your localhost

The notebooks are located in the `\app` directory. Once navigated there,
each notebook can be accessed and ran individually. More can also be made
and the data is free to be interacted with.

### Some Information on Quarto Commands

Renewable ML uses [Quarto](https://quarto.org/) for the creation of the
website. Detailed steps can always be found at [their documentation](https://quarto.org/docs/get-started/).
for how to get started with Quarto. The process of creating
the website for this repository was a followed (assuming VS Code is being used):

Be sure to have cloned/forked the repo and that Python 3.11.X and poetry are installed.

- First, install the [Quarto CLI](https://quarto.org/docs/get-started/)
- Then get the Quarto extension from the **Extensions** tab in VS Code.
- Now click on the search bar up at the top of VS Code and type
    `>Quarto: Create Project`
- Select that command and click on `Website Project`
- Choose a directory or make a new one to house the website project.
- Navigate to this new directory in the CLI
- Type `poetry shell` and then `poetry install`
- Now commands like `quarto preview` and `quarto render` can be used
    to update a local version of the website and track changes.
- The command `quarto publish netlify` was used for this project to host the website
    on [Netlify](https://www.netlify.com/)

One also has the option boot up the Renewable ML website locally just by using
`quarto preview app` in the repository's root in the CLI.

There is now a website that hosts the note books!
You can find it [here](https://aidanneeson610artifact.netlify.app/)

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
of two main data sets: `wind.csv` and `solar.csv`. These two data sets
each consist of 13 attributes, with each attribute having either approximately
125000 or 11500 observations depending on the data set being examined. Full
diagnostic reports regarding these data sets can be found on the
[website](https://aidanneeson610artifact.netlify.app/data.html).

These data sets are the heart of this artifact, in that they are what drives
the predictions to be made. Making use of Python Jupyter Notebooks, the data
are read in as Pandas data frames and passed into various Scikit-Learn
machine learning tools. These include random forest, support vector machine,
and artificial nerual network. These tools, along with Matplotlib,
allow for drawing connections between the data through graphical
representations.

## Future Plans

Some future plans include expanding this work into a user study.
An idea that was brought up early in development consisted of using the
models generated through this process to create a tool that users can
interface with that allows for them to make predictions regarding
renewable energy array simply by inputting the location of the array,
as well as the desired size of the array. Their opinions of the process
would then be extracted and compared to their experience using similar,
more complex tools like [PVsyst](https://www.pvsyst.com/). The hope would
be to track that users have an easier time and gain better insights
relative to time spent on the tool when interacting the tool produced
from this work.

Other avenues for future work include better data collection, model
fine-tuning, and using a larger selection of models.
