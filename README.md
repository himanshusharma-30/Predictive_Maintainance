# Predictive_Maintainance

A major problem faced by businesses in asset-heavy industries such as manufacturing is the significant costs that are associated with delays in the production process due to mechanical problems. Most of these businesses are interested in predicting these problems in advance so that they can proactively prevent the problems before they occur which will reduce the costly impact caused by downtime. 

This example brings together common data elements observed among many predictive maintenance use cases and the data itself is created by data simulation methods.

The business problem for this example is about predicting problems caused by component failures such that the question "What is the probability that a machine will fail in the near future due to a failure of a certain component?" can be answered. 

The purpose of the project is to build a classification model using the 'Predictive maintenance' dataset, which consists of 10 000 data points stored as rows with 8 features in columns. The classifier will have to be able to predict the target variable, which takes value '0' if the machine has no failure and therefore no maintenance is needed, '1' if on the contrary, if some kind of damage has been revealed in the machine and it needs maintenance.
