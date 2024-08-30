---
layout: post
title: "Data analytics part 1"
date: 2024-08-27 01:43:18 +0530
categories: Deep Learning
---


* Random Variable * 
A random variable is a function that associates a real number with each element in the sample space.
a probability distribution is the mathematical function that gives the probabilities of occurrence of possible outcomes for an experiment

* Discrete Probability Distribution *

Binomial Distribution

The binomial distribution is a discrete probability distribution that describes the number of successes in a fixed number of independent and identically distributed Bernoulli trials.
A Bernoulli trial is a random experiment where there are only two possible outcomes
- Success( with probability p)
- Failure (with probability 1-p)
![DA](/assets/da_p1.jpg)
mean = np
variance = npq

Hypergeometric Distribution

Necessary condition for binomial distribution is all events are independent of each other but in hypergeometric distribution it does not require to be independent and is based sampling without replacement
If we randomly select n items without replacement from a set of  N items of which m of the item are of one type and N - m of the items are of a second type then the probability mass function of the discrete random variable  X is called the hypergeometric distribution and is of the form:

![DA](/assets/da_p2.jpg)
where the support S is the collection of nonnegative integers x that satisfies the inequalities: x<=n , x<=m , n-x<= N-m

mean = nk/N
variance = nk(N-k)(N-n)/(N^2 * (N-1))

Poison Distribution 

When you have a large number of events with a small probability of occurrence, then the distribution of number of events that occur in a fixed time interval approximately follows a Poisson distribution.
Mathematically speaking, when n tends to infinity (n→ infinity) and the probability p tends to zero (p→ 0) the Binomial distribution can approximated to the Poisson distribution.
![DA](/assets/da_p3.jpg)
 " λ  " represents lambda, which is the expected number of possible occurrences. It is also sometimes called the rate parameter or event rate, and is calculated as follows: events/time * time period.
 mean = λ*t
 variance = λ*t

Discrete uniform distribution
discrete uniform distribution is a symmetric probability distribution wherein a finite number of values are equally likely to be observed; every one of n values has equal probability 1/n. 

mean = (b+a)/2
variance = ((b-a+1)^2 -1) /12


