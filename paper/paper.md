---
title: '`MoRSAIK`: Sequence Motif Reactor Simulation, Analysis and Inference Kit in Python'
tags:
  - Python
  - abiogenesis
  - sequence motif
  - dynamics
  - RNA
  - DNA
  - proteins
authors:
  - name: Johannes Harth-Kitzerow
    orcid: 0000-0001-5864-2258
    affiliation: [1,2,3,4]
  - name: Ulrich Gerland
    orcid: 0000-0002-0859-6422
    affiliation: [3,4]
  - name: Torsten A. Enßlin
    orcid: 0000-0001-5246-1624
    affiliation: [1,2,4]
affiliations:
 - name: Max-Planck-Institut für Astrophysik, Karl-Schwarzschild-Str. 1, 85748 Garching, Germany
   index: 1
 - name: Ludwig-Maximilians-Universität München, Geschwister-Scholl-Platz 1, 80539 Munich, Germany
   index: 2
 - name: Technische Universität München, James-Franck-Str. 1, 85748 Garching, Germany
   index: 3
 - name: Exzellenzcluster ORIGINS, Boltzmannstr. 2, 85748 Garching, Germany
   index: 4
date: 02 December 2025
bibliography: paper.bib
header-includes: 
  - \usepackage{tikz}
---

# Statement of need 

Origins of life research investigates how life could emerge from prebiotic chemistry only.
Living systems as we know them today rely on RNA, DNA and proteins.
According to the central dogma of molecular biology, information is stored in DNA,
transferred by RNA resulting in proteins that catalyze functional reactions, such as synthesis and replication of DNA and RNA. <!--- https://libgallery.cshl.edu/items/show/52220 --->
One possible explanation of how this mechanism evolved provides the RNA world hypothesis [@Crick1968TheOrigin; @Higgs2015TheRNAWorld; @Orgel1968Evolution; @Pressman2015TheRNAWorld; @Szostak2012TheEightfold].
It states that life could emerge from RNA strands only, storing and transferring biological information,
as well as catalyzing reactions as ribozymes.
Before this state could have emerged, however,
the prebiotic world was probably a purely chemical pool of short RNA strands with random sequences and without biological function.
Despite the lack of guidence by proteins, the RNA sequences reacted with each other.
In such an RNA reactor RNA strands perform hybridization and dehybridization, as well as ligation and cleavage.
In this context relevant questions are
what are the conditions that allow longer RNA strands to be built and how can information carrying in RNA sequence emerge?

A key reaction for the emergence of longer RNA strands is templated ligation.
There, two strands hybridize adjacent onto a template strand and ligate.
The rate of this reaction is the larger, the better the two strands match the complementary sequence of the template strand.
The extended strands can then serve as a template for the next generation of templated ligation.
This leads to an acceleration of production of complementary strands.
This process, however, is highly sensitive to environmental conditions determining the reaction rates within an RNA reactor [@Goeppel2022Thermodynamic; @Rosenberger2021SelfAssembly].

In order to investigate those RNA reactors, efficient simulations are needed
because the space of possible RNA sequences increases exponentially with the length of the strands,
as well as the number of reactions between two strands.
In addition, simulations have to be compared to experimental data for validation and parameter calibration.
Here, we present the ``MoRSAIK`` python package for sequence motif (or k-mer) reactor simulation, analysis and inference.
It enables users to simulate RNA sequence motif dynamics in the mean field approximation
as well as to infer the reaction parameters from data
with Bayesian methods and to analyze results by computing observables and plotting.
``MoRSAIK`` simulates an RNA reactor by following the reactions and the concentrations of all strands inside up to a certain length (of four nucleotides by default).
Longer strands are followed indirectly, by tracking the concentrations of their containing sequence motifs of that maximum length.

# Summary

\begin{figure}
\center
\begin{tikzpicture}[
        modulenode0/.style={double,rounded corners, draw=black!60, fill=white!5, very thick, minimum size=3mm},
        modulenode1/.style={double,rounded corners, draw=black!60, fill=white!5, thick, minimum size=3mm},
        objnode1/.style={ellipse, draw=black!60, fill=white!5, very thick, minimum size=3mm},
        opnode1/.style={rectangle, draw=black!60, fill=white!5, very thick, minimum size=3mm},
    ]
    %Nodes
    \node[modulenode0]  (morsaiknode) {morsaik};
    \node[modulenode1]  (domnode)  [below of= morsaiknode] {domains};
    \node[modulenode1]  (plotnode)  [right of= domnode, xshift=.5cm] {plot};
    \node[modulenode1]  (infernode)  [above of= morsaiknode] {infer};
    \node[modulenode1]  (readnode)  [left of= infernode, xshift=-.5cm] {read};
    \node[modulenode1]  (utilsnode)  [left of= domnode, xshift=-.5cm] {utils};
    \node[modulenode1]  (getnode)  [right of= infernode, xshift=.5cm] {get};
    %Lines
    \draw[-, very thick] (morsaiknode.south) -- (domnode.north);
    \draw[-, very thick] (morsaiknode.south east) -- (plotnode.north);
    \draw[-, very thick] (morsaiknode.north west) -- (readnode.south);
    \draw[-, very thick] (morsaiknode.north) -- (infernode.south);
    \draw[-, very thick] (morsaiknode.north east) -- (getnode.south);
    \draw[-, very thick] (morsaiknode.south west) -- (utilsnode.north);
\end{tikzpicture}
\caption{Overview of the MoRSAIK package modules.}
\end{figure}

The core of the ``MoRSAIK``-package are chemical rate equations simulating the reaction dynamics inside an RNA strand reactor.
$$\dot{c}_p = f_p^{(\text{ext})} + f_p^{(\text{cut})}$$
The key reaction rates are templated ligation rates, $f_p^{(\text{ext})}$, and breakage rates, $f_p^{(\text{cut})}$, for the motif $p$,
where "ext" refers to the extension of strands by the creation of motifs and "cut" for the destruction of motifs due to cutting of strands.
The terms for templated ligation reaction rates take the form
$$f_p^{(\text{ext})} = k_\text{ext}(p,l,r,t) c_l c_r c_t,$$
with the concentrations $c_i$ of the two reactants $i=l$ and $i=r$ that form the left and right part of the product, respectively, the template $i=t$, and the produced motif $i=p$.
The extension rate constants $k_\text{ext}$ model templated ligation in a hybridization-dehybridization equilibrium for resulting motifs or strands $p$.
For the reactants ($p=l$ or $p=r$) the same equations apply with negative reaction rate constants.
The breakage terms take the form
$$f_p^{(\text{cut})} = k_\text{cut}(p,b) c_b,$$
with breaking reactant $i=b$ and products $p=l,r$, as well as a negative rate for the reactant $p=b$.
For mathematical details, please see [@HarthKitzerow2024Sequence] or the documentation.
With ``MoRSAIK`` one can simulate motif concentration trajectories given the reaction rate constants,
infer the reaction rate constants given templated ligation counts,
as well as compute the reaction rate constants based on the free energy model of [@Goeppel2022Thermodynamic] and its adaptation for motifs [@HarthKitzerow2024Sequence].

The ``MoRSAIK``-package consists of six modules: `domains`, `read`, `infer`, `get`, `utils`, `plot`, and the objects themselves (`obj`).
The `domains`-module extends (classic) `nifty8` domains [@nifty5] to store the
`unit` [@units] additionally to the shape `obj`ects to ensure consistency.
The `read`-module contains a set of functions that read `yaml` files [@pyyaml], saved arrays, and different data files such as output and parameter files of RNA strand reactor simulations.
The `infer`-module contains all functions that compute observables from parameters or data.
The `get`-module either reads in a stored result object if found in the result archive ``MoRSAIK`` maintains or triggers the generation via the `infer`-module.

The `utils`-module is a set of different useful functions that do not belong to one of the other modules.
The `plot`-module provides a set of plotting functions for several ``MoRSAIK``-objects.
It is based on `matplotlib` [@Hunter2007Matplotlib].
``MoRSAIK``-objects are implemented in the `morsaik/obj` directory, which is not a separate (sub)module, but imported directly with ``MoRSAIK``.
Typical objects are motif vectors, which are arrays of motif concentrations motif trajectories, time trajectories of motifs vectors, and motif trajectory ensembles, which are ensembles of motif trajectories.
For readability, objects are designed as `namedtuple`s on the user level.
During computation they are transformed to ``Jax``-arrays [@Jax2018Github] to ensure efficient computations and differentiability, where needed.
All functions are split into small subfunctions to ensure flexibility.
For the inference, we use Geometric Variational Inference [@Frank2021GeoVI;
@Knollmueller2020Metric]
implemented in ``nifty8.re`` [@niftyre; @nifty5; @nifty3; @nifty1].
For integration of ordinary differential equations, we use ``diffrax``
[@Kidger2021OnNeural] and ``scipy.integrate.solve_ivp`` [@Virtanen2020SciPy].
The implementation in ``Jax`` enables fast computation and differentiable models for inference from data with ``NIFTy``.
``MoRSAIK`` is the first package that provides implementation of Bayesian inference methods for RNA reactor simulations.

Current research projects using ``MoRSAIK`` are the comparison of the RNA sequence motif dynamics in an RNA strand reactor to the dynamics in an RNA motif reactor [@HarthKitzerow2024Sequence]
and the comparison of the inferred motif dynamics to the originating strand dynamics [@HarthKitzerow2024Projection].

# Acknowledgements

We thank Jakob Roth, Gordian Edenhofer, Philipp Frank, Martin Reineke, Viktoria Kainz, Tobias Göppel, Ludwig Burger, Julio C. Espinoza Campos, Julius Lehmann and Paul Nemec for stimulating discussions.
The project was financially supported by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's Excellence Strategy – EXC-2094 – 390783311.

# References
