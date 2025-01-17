---
title: "Realistic Synthetic Social Networks with Graph Neural Networks"
collection: publications
permalink: /publication/resonet
excerpt: 'Social network analysis faces profound difficulties in sharing data between researchers due to privacy and security concerns. A potential remedy to this issue are synthetic networks, that closely resemble their real counterparts, but can be freely distributed. generating synthetic networks requires the creation of network topologies that, in application, function as realistically as possible. Widely applied models are currently rule-based and can struggle to reproduce structural dynamics. Lead by recent developments in Graph Neural Network (GNN) models for network generation we evaluate the potential of GNNs for synthetic social networks. Our GNN use is specifically within a reasonable use-case and includes empirical evaluation using Maximum Mean Discrepancy (MMD). We include social network specific measurements which allow evaluation of how realistically synthetic networks behave in typical social network analysis applications.
We find that the Gated Recurrent Attention Network (GRAN) extends well to social networks, and in comparison to a benchmark popular rule-based generation Recursive-MATrix (R-MAT) method, is better able to replicate realistic structural dynamics. We find that GRAN is more computationally costly than R-MAT, but is not excessively costly to employ, so would be effective for researchers seeking to create datasets of synthetic social networks.'
date: 2022-12-15
venue: 'arXiv'
paperurl: 'http://neutralpronoun.github.io/alexowendavies.github.io/resonet-page.pdf'
---

Abstract
======

Social network analysis faces profound difficulties in sharing data between researchers due to privacy and security concerns.
A potential remedy to this issue are synthetic networks, that closely resemble their real counterparts, but can be freely distributed.
Generating synthetic networks requires the creation of network topologies that, in application, function as realistically as possible.
Widely applied models are currently rule-based and can struggle to reproduce structural dynamics.
Lead by recent developments in Graph Neural Network (GNN) models for network generation we evaluate the potential of GNNs for synthetic social networks.
Our GNN use is specifically within a reasonable use-case and includes empirical evaluation using Maximum Mean Discrepancy (MMD).
We include social network specific measurements which allow evaluation of how realistically synthetic networks behave in typical social network analysis applications.
We find that the Gated Recurrent Attention Network (GRAN) extends well to social networks, and in comparison to a benchmark popular rule-based generation Recursive-MATrix (R-MAT) method, is better able to replicate realistic structural dynamics.
We find that GRAN is more computationally costly than R-MAT, but is not excessively costly to employ, so would be effective for researchers seeking to create datasets of synthetic social networks.

Informally
======

Fake social network graphs would be great, because they wouldn't contain real people, but would allow useful research.
People currently used rule-based models (fit a distribution, follow N set steps recursively to make a graph) to produce social networks.
GNNs are more recent generators, and are doing very well on molecules.
Here we show that they also do better on social networks.

[Download paper here](https://neutralpronoun.github.io/alexowendavies.github.io/files/resonet-page.pdf)