# Covid-19 prediction
### Introduction 
* Simulating covid-19 pandemic and trying to grab future outcomes, like 
  * When the pandemic would end
  * Total no. of deaths
  * Total no. of cases, etc.
  
### Python Libraries used 
* **pandas** 
  * It is a python library which provides high-performance and easy-to-use data structures and **data manipulation and analysis** tools.
  * It mainly provides three form of data_structure: **series** , **dataFrame**(2d array) and **panel** (3d array) . These are build over numpy array hence really fast.
* **matplotlib** 
  * It is a python library used for **data-visulization**
* **scikit-learn** 
  * It provides efficient **tools for machine learning** including classification, regression and clustering
  * It's build over numpy, scipy and matplotlib

### Working
* We need to find series of data that provides the most reliable representation of the epidemic progression.
* In order to do this , let's visualize the data collected in the form of csv file from official site of government of Italy.
  * date vs **new positive cases**.
    * Epidemic seemed to slow down on weekends
    * Taking **moving average** of window size 7 days to smoothen the data.
    * It has bias over no of test performed.
  * date vs **percentage of new positive cases to no of test performed**
    * This provides a more reliable representation of the **epidemic progression**.
    * There are still a lot of bias as follow: 
      * Some patients can also be tested multiple times, and there is no way to detect this from our dataset.
      * Testing procedures have changed considerably over time, ranging from testing only severe symptomatic patients to mass testing of entire populations in specific districts
  * date vs **intensive care**
    * Seems to follow a more regular trend
      * It could be that this number is lowering as a consequence of an increasing number of daily deaths.
  * ASSUMPTION : the combined value of **intensive care** and **daily deaths** can be a reliable variable for estimating the current epidemic progression, and for modeling future trends. Let **distress** = **intensive care** + **daily deaths**
  * date vs **distress**
    * This could be the perfect variable for estimating the current epidemic progression.
  * Modeling the epidemic trend
    * We now can build a Linear Regression model and train it with the data of distress from that date, April 1st. 
    * We find **coefficient of determination R²** of the prediction. R² will give some information about the goodness of fit of a model. 
  * plot the known and predicted data together with the forecast and the error lines.
### Conclusions
Our model tells us that the COVID-19 epidemic will end in Italy between **26th May 2020 and 22nd July 2022**, if the current trend is maintained over time. The current model can not take into account unexpected changes in the system, such as the gradual loosening of lockdown restrictions, or the effects of warmer temperature over the virus spread.
### References 
* supervised/unsupervised learning : https://machinelearningmastery.com/supervised-and-unsupervised-machine-learning-algorithms/
* linear regresion : https://machinelearningmastery.com/linear-regression-for-machine-learning/
* Itly covid-19 data : https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv
* TowardsDataScience article : https://towardsdatascience.com/model-the-covid-19-epidemic-in-detail-with-python-98f0d13f3a0e

