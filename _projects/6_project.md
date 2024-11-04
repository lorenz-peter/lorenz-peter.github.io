---
layout: post
title: Research Trends
date: 2023-03-20 14:37:00-0400
description: an example of how to use Bootstrap Tables
tags: research
categories: research
giscus_comments: true
related_posts: true
pretty_table: true
---

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/load_json.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/load_json.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
{% jupyter_notebook jupyter_path %}
{% else %}

<p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}