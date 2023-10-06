---
title: "Hierarchical GNNs for Large Graph Generation"
collection: publications
permalink: /publication/resonet
excerpt: 'Large graphs are present in a variety of domains, including social networks, civil infrastructure, and the physical sciences to name a few. Graph generation is similarly widespread, with applications in drug discovery, network analysis and synthetic datasets among others. While GNN (Graph Neural Network) models have been applied in these domains their high in-memory costs restrict them to small graphs. Conversely less costly rule-based methods struggle to reproduce complex structures. We propose HIGGS (Hierarchical Generation of Graphs) as a model-agnostic framework of producing large graphs with realistic local structures. HIGGS uses GNN models with conditional generation capabilities to sample graphs in hierarchies of resolution. As a result HIGGS has the capacity to extend the scale of generated graphs from a given GNN model by quadratic order. As a demonstration we implement HIGGS using DiGress, a recent graph-diffusion model, including a novel edge-predictive-diffusion variant edge-DiGress. We use this implementation to generate categorically attributed graphs with tens of thousands of nodes. These HIGGS generated graphs are far larger than any previously produced using GNNs. Despite this jump in scale we demonstrate that the graphs produced by HIGGS are, on the local scale, more realistic than those from the rule-based model BTER. '
date: 2023-06-20
venue: 'arXiv'
paperurl: 'https://neutralpronoun.github.io/alexowendavies.github.io/files/resonet-page.pdf'
---

Abstract
======

Large graphs are present in a variety of domains, including social networks, civil infrastructure, and the physical sciences to name a few.
Graph generation is similarly widespread, with applications in drug discovery, network analysis and synthetic datasets among others.
While GNN (Graph Neural Network) models have been applied in these domains their high in-memory costs restrict them to small graphs.
Conversely less costly rule-based methods struggle to reproduce complex structures.
We propose HIGGS (Hierarchical Generation of Graphs) as a model-agnostic framework of producing large graphs with realistic local structures.
HIGGS uses GNN models with conditional generation capabilities to sample graphs in hierarchies of resolution.
As a result HIGGS has the capacity to extend the scale of generated graphs from a given GNN model by quadratic order.
As a demonstration we implement HIGGS using DiGress, a recent graph-diffusion model, including a novel edge-predictive-diffusion variant edge-DiGress.
We use this implementation to generate categorically attributed graphs with tens of thousands of nodes.
These HIGGS generated graphs are far larger than any previously produced using GNNs.
Despite this jump in scale we demonstrate that the graphs produced by HIGGS are, on the local scale, more realistic than those from the rule-based model BTER.

Informally
======

Most models for generating graphs are super expensive in-memory, because they have to consider the possibility of an edge between every pair of nodes.
Here we use models with conditional generation (*"generate me a graph like this please"*) to produce much bigger graphs in hierarchies.
First one model produces a low resolution template, where each node represents a subgraph.
Then a second model produces those subgraphs, conditioned on the features of the nodes in the previous one.
Finally a third model produces the high-resolution edges between these subgraphs.
We make really big graphs (>20k nodes!) which are much more realistic on the zoomed-in hundreds-of-nodes scale.
We ran into issues with metrics, as the currently used set aren't very expressive, but visualisations show what we're doing is along the right lines.


[Download paper here](https://neutralpronoun.github.io/alexowendavies.github.io/files/resonet-page.pdf)