---
layout: post
title: Adversrial Examples (Related Work)
description: shows latest related work
importance: 3
category: paperlist
---

## A complete list of all (arXiv) adversarial papers (under construction)

by Peter Lorenz

### Description

Staying current with the latest research in adversarial examples can be a daunting task, given the rapid increase in publications each year. I've dedicated myself to meticulously tracking these papers over the past few years and realized that sharing this curated list could benefit others.

The sole criterion for selecting papers for this list is their primary focus on adversarial examples or extensive use of them. However, due to the sheer volume of papers, I can't guarantee that I've captured every single one.

Nevertheless, I've made every effort to include papers that predominantly discuss adversarial examples. Occasionally, there may be papers that do not strictly adhere to these criteria or where judgment calls were inconsistent. If you notice any discrepancies, please reach out via email, and I'll promptly address them.

As a result, this list remains unfiltered, encompassing any paper primarily centered around adversarial examples without passing judgment on their quality. For a curated selection of what I consider excellent and worthwhile reads, please refer to the Adversarial Machine Learning Reading List.

A note on the data: This list updates automatically with new papers, sometimes before I can manually review them. I typically conduct this review twice a week to remove papers unrelated to adversarial examples. Therefore, there might be occasional false positives among the most recent entries. Each new, unverified entry includes a probability score from my simple yet well-calibrated bag-of-words classifier, indicating its likelihood of focusing on adversarial examples.

Below, you'll find the comprehensive paper list. I've also provided [JSON file](https://github.com/lorenz-peter/lorenz-peter.github.io/blob/master/assets/json/model_stealing_papers.json) [guide](https://lorenz-peter.github.io/blog/2024/load-json/) containing the same data, including one with abstracts. If you use this data for any interesting projects, I'd love to hear about your experiences.

Recently, another website was deployed to discover research trends, [researchtrend.ai](https://researchtrend.ai/communities/AAML).

In future, I might plan to to add also papers from [eprint.iacr.org](https://eprint.iacr.org/).

## Acknowledgment

The idea is derived from Nicolas Carlini:
[nicholas.carlini.com/writing/2019/all-adversarial-example-papers.html](https://nicholas.carlini.com/writing/2019/all-adversarial-example-papers.html).

## Table

<table
  data-toggle="table"
  data-show-fullscreen="true"
  data-pagination="false"
  data-search="true"
  data-show-columns="true"
  data-url="{{ '/assets/json/model_stealing_papers.json' | relative_url }}">
  <thead>
    <tr class="tr-class-1">
      <th data-field="date" data-sortable="true" data-width="50">date</th>
      <th data-field="title" data-sortable="true" data-formatter="addLink">title</th>
      <th data-field="author" data-sortable="true" >author(s)</th>
      <th data-field="link" data-visible="false">Link</th>
    </tr>
  </thead>
</table>

<script>
  function addLink(value, row) {
    // Access the 'link' property directly from the row object
    var link = row.link;
    // Create an anchor tag with the link and value
    return `<a href="${link}" target="_blank">${value}</a>`;
  }
</script>

