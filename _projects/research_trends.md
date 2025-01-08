---
layout: page
title: Research Trends
description: an example of how to use Bootstrap Tables
importance: 3
category: paperlist
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