---
title = "Microservices Part 1"
date = "2025-08-19T22:28:21+05:30"

# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = ["distributed_systems","notes",]
---


# Chapter 1
In large codebased it's become very difficult to add new features or even understand the codebase. It's very difficult to understand where changes need to be made as code related to similar function is all over the place.

Microservices are small autonomous parts that works together. We need to break down the monolithic structure untill codebases can be handle by small teams.
Microservices need to be independent we can be able to make some changes in a microservice without changing some other microservice.

Pros:
- we are able to choose right technology for each microservices.
- if one component fails rest of the system can carry on.
- we can scale those services that need scalling.
- ease of deployement
- composability