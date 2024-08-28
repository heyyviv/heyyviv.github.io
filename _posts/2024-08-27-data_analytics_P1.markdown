---
layout: post
title: "Data analytics part 1"
date: 2024-08-27 01:43:18 +0530
categories: Deep Learning
---

Set of Data is collection of observed values representing one or more characteristics of some objects or unit.

Population is entire dataset representing the entire entities of interest.
Example all TV viewer in a country/world.
Sample is a data set consisting of a population. Subset of Population.
Students in a school is population and student in class 12 is sample.
Statistics is a quantity calculated from data that describe a characteristic of sample.
Statistical inference is a process of using sample statistics to make decision about population.

Data Summarization is used
- to identify typical characteristics of data( to have an overall picture)
- to idetify which data should be treated as noise or outlier

*Measures of Location*
measuring of central tendency

Distributive measure
The distributive measure can be computed for a given set of data by partitioning the data into smaller subsets, computing the measures for each subset and then merging the results in order to arrive at the measure's value for the entire data
example -> sum,count

Algebraic measire
It is a measure that can be computed by applying an algebric function to one or more distributive measures

average -> sum()/count()

Holistic measure
It is a measure that must be applied to entire dataset as whole
example -> meadian

*Types of  Mean*
Arithmetic Mean -> 1/n(x1+x2+ .. xn)
Weighted Mean  -> each sample value xi is associated with a weight wi for i=1,2,3...n
WM = (w1x1+w2x2 ..+wnxn)/(w1+w2+..wn)
Trimmed mean -> if there is extreme values in a sample , mean is infuenced heavily by those values.So in this mean is obtained after chopping off values at the high and low extremes.


set = {x1,x2}
arithmetic mean = (x1+x2)/2
Harmonic mean = 2/(1/x1+1/x2)
