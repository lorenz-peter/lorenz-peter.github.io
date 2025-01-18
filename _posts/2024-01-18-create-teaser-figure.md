---
layout: post
title: How do you make your own illustrations for your paper?
date: 2024-01-18 16:40:16
description: conference
tags: paper, inkscape 
categories: writing
giscus_comments: true
---

#  Teaser Image

This is a central question, when most experiments are done and the story of the paper is becoming clearer and clearer. 
Then the question arises how to support the story visually? 

There is the option of the [teaser image](https://academia.stackexchange.com/questions/44837/what-is-teaser-image), that is a visually striking and representative image that is used to grab the reader's attention and provide a quick visual summary of the paper's content. 
It is typically placed prominently, often near the title or abstract, to entice readers to learn more.


## Tools

There are several tools to create an teaser image, which will be discussed in this article. 
The most universal tool is inkscape and can help you in the most situations. 

Nevertheless, from which tool the graphic comes from, the final format should be in `.png', which works mostly most smoothly in LaTeX.

## Powerpoint or Draw.io

Powerpoint drawing can be useful to sketch very fast on high level a process. 
Powerpoint can export slides in PDF format, which can be edited by next tool Inkscape. 

As an alternative, [draw.io](www.draw.io) can be a very usfuel tool to draw such flow diagrams and show relations between components. 

## Inkscape

Inkscape is an open-source tool for vector graphics. 
Every file format in `.svg` or `.pdf` can be imported into Inkscape.

My favourite function at incscape is [resize page to drawing or selection](https://imagy.app/inkscape-fit-page-to-selection/), which does not leave any edges at the corner and then it can be exported in different formats. 



## LaTeX Drawing Format

TiKZ is a LaTeX package that allows you to create high-quality graphics, diagrams, and illustrations directly within your LaTeX document. It is a powerful tool for creating vector-based graphics such as geometric figures, plots, and other illustrations.

However, writing line for line can be exhausting. There are plenty of tools online, which can create from a vector graphic to LaTeX code, e.g. 

 - [GeoGebra](https://cs.overleaf.com/learn/latex_LaTeX_Graphics_using_TikZ%3A_A_Tutorial_for_Beginners_(Part_2)%E2%80%94Generating_TikZ_Code_from_GeoGebra)
 - [TikzMaker](https://tikzmaker.com)
 - [Q Uiver](https://q.uiver.app).