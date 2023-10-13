---
title: 'Graph Generation Papers'
date: 2023-11-13
bibliography: references.bib
permalink: /posts/2023/11/graph-generation/
tags:
  - graphs
---

Here is essentially a big list of graph generation papers.
It might be useful to someone, I just haven't put it together into a proper review yet.


---

# Review Protocol

Background: Rationale for Survey

:   Graph generation surveys thus-far broadly carry the lens of
    application - that is, presentation is "X problem is addressed by
    the following, Y, Z\...". Instead, a survey of papers that focuses
    on broad capabilities, for example features and scale, would allow
    easier identification of future research areas.

Research Questions

:   What are the current capabilities of graph generation methods? How
    have these capabilities developed over time, and which technologies
    or algorithms have aided this development? What can we expect from
    future graph generation methods?

Search Strategy

:   Identify works initially through existing surveys. Following this
    body of works, search through broad keywords on online platforms, eg
    "graph generation, network generation, molecule generation". Finally
    identify works through conference and journal archives. Focus should
    be on GNN methods, but with some constrast to rule-based methods.
    Non-archival papers (ie from ArXiv) older than a set time period
    without publication should be excluded.

## Broad Application Areas

Molecule Generation

:   Does what it says on the tin. Often is very targetted, ie "generate
    a molecule with X property" or "generate a molecule with Y
    interaction with Z other molecule". As such these are mostly
    conditional models, almost all with node labels. Metrics often focus
    on validity ("is this possible?") and synthetic accessibility ("can
    we actually make this?").

General Graphs

:   Non-domain specific models targetted at generating graphs in
    general, however often use domain specific datasets, such as the
    chemical datasets QM9 and Zinc.

Scene Generation

:   The task of mapping an image into a semantic scene graph. This
    requires the correct identification of objects and how they are
    related.

# Big list of graph generation papers

## Rule-Based

-   @Erdos1960ONGRAPHS propose one of the earliest graph generators, but
    as a formulation of random graphs. For each node-node pair,
    $v_i, v_j$ there is a set probability of an edge between them $P_e$.
    The work itself is a contrast between viewing this as an
    evolutionary process, choosing edges one-by-one, and in simply
    sampling the whole graph at once.

-   @Chakrabarti2004R-MAT:Mining propose probably the best-known
    graph-generation model R-MAT. It recursively sub-divides an
    adjacency matrix into quadrants, with a set edge probability
    $a + b + c + d = 1$ for each quadrant.

-   @Leskovec2005RealisticMultiplication use Kronecker products to
    produce self-similar graphs. Can handle temporal data, but seems to
    be just producing a new graph for each timestep, and without
    attributes.

-   @Conti2011AGraphs use modelled social closeness and invested time to
    construct ego networks.

-   @Seshadhri2012CommunityGraphs propose BTER, an ER (Erdos-Renyi)
    graph of ER graphs. This aims to give generated graphs community
    structure. By iterates over an input degree distribution, with an ER
    graph constructed from each unique degree value (with some extra
    steps). This means it requires a whole input degree distribution. A
    follow-up (?) paper @Kolda2014AStructure provides a scalable
    implementation, up to 4.6 billion edges.

-   @Ali2014SyntheticData use @Yang2017LeveragingNetworks's graph
    generator as a benchmark. This work alternates adding edges and
    nodes depending on the target density of the resulting network. Edge
    attachment is based on homophily, ie more similar nodes are more
    likely to share an edge.

-   @Nettleton2016AGraphs use R-MAT to produce topologies. They then
    distribute features on a per-case basis, using existing information,
    although in theory this framework supports using a more intelligent
    learnt process, as they do consider $P(X|Y)$ for various attributes.
    These attributes are distributed in communities to ensure homophily.
    *NB: This is a very long work\...*

-   @Jensen2019ASpace propose a new Graph-Based Genetic-Algorithm
    (GB-GA) to explore chemical space, optimising $\log(P)$ and
    synthetic accessibility. It works on much smaller datasets than ML
    approaches, and uses a Monte Carlo Tree Search (MCTS) algorithm.

-   @Akoglu2008RTM:Graphs propose RTG, a recursive model to produce
    time-evolving graphs based on random typing.

-   @Akoglu2008RTM:Graphs first identify patterns in real evolving
    graphs, then propose a generator to produce graphs with the patterns
    they identified. This model is called the Recursive Tensor Model
    (RTM), and works almost entirely through recursive multiplication.

-   @Khan2018OnlineAttachment propose what looks like a typical
    pipeline: Create a dataset of user information, then compute
    similarity between users, then use that similarity to connect edges.

-   @Sagduyu2018SyntheticGeneration present an end-to-end social network
    generation tool SHIELD. Topology is generated through whichever
    rule-based model performs best. This is more of a composite
    framework of models from other authors. Generation of attributes is
    handles through node class and identified communities. Textual
    generation is primarily through Markov chains, and tweet attributes
    beyond text refer back to the original dataset.

-   @Robins2007RecentNetworks reviews specifications for "exponential
    random graph" models, and demonstrate that these p\* methods avoid
    the near-degeneracy problem experienced with Markov random graphs.
    p\* methods are summarised in @Robins2007AnNetworks.

-   @Edunov2016Darwini:Graphs propose Darwini, a three stage model for
    generating synthetic social graphs given an input graph. In the
    first stage, each node is assigned a target degree and clustering
    coefficient based on the original graph. In the second stage nodes
    are grouped into smaller communities and edges are added such that
    the target clustering and degree values are roughly matched. Lastly
    nodes are connected across communities to match the actual target
    distributions. This is very much like @Seshadhri2012CommunityGraphs.

-   @Kim2012MultiplicativeNetworks propose the Multiplicative Attribute
    Graphs (MAG) model, a framework for considering how a discrete set
    of node attributes can be considered in the context of the graph
    structure. They parameterise through the link-affinity matrix, which
    describes how the value of a particular attribute affects the
    probability of a link between a pair of nodes. They show that this
    formulation gives rise to the giant connected components often seen
    in real social networks.

-   @Pham2013S3G2:Generator propose Scalable Structure-correlated Social
    Graph Generator (S3G2). S3G2 generates edges between nodes
    correlated against their attributes.

-   @Penschuck2020RecentGeneration explore recent advances in random
    graph models for generation, with a focus on generating massive
    graphs.

## Simulation

-   @Barrett2009GenerationNetworks use simulated agents to construct
    large-scale (hundreds of million nodes) contact networks. This
    includes features for each node, which are constructed prior to
    network synthesis. This is a highly domain specific work, and as
    such they employ domain-specific quality metrics.

-   @Lin2011SimulatingXplore simulate sharing behaviours on social
    networks. There is a parameter $r_{i,j}$ for each user pair, which
    determines the strength of friendship, which is almost an edge
    equivalent - but as all users share such $r$, this is a complete
    weighted graph, so consideration isn't really of graph structure.

-   @DeCaux2014DynamicInteractions simulate multiple agents moving on a
    surface. If another agent is in range, defined by a function, there
    is considered to be an edge between them.

-   @Bernstein2013StochasticNetworks use an agent-based simulation to
    generate activity data with "narrative power" while providing
    statistical diversity. Their generator is a three step process of
    random drawing: 1. draw the time at which the agent's event
    occurs 2. draw the role the agent adopts at that time 3. draw the
    action the agent actually takes.

## Quantum!

-   @Yan2023QuantumEmbedding propose to use a Paramaterised Quantum
    Circuit (PQC) to encode molecules. The 3D coordinates of nodes
    (atoms) are converted to rotation and torsion angles, then encoded
    into qubits. The PQC is applied to serve as trainable layers, and
    the output is adopted into final node embeddings. They demonstrate
    good performance with very few parameters on property prediction and
    geometry generation. They also show its potential to be applied on a
    real quantum device.

## Deep-Learning (non-GNN)

-   @Gu2019SceneReconstruction address the problem of scene generation
    using external knowledge and image reconstruction. They construct a
    graph using object and relationship classification, with external
    knowledge included to optimise these classifications.

-   @Li2017SceneCaptions address the problem of scene generation using
    captions. They refine their scene graphs using message passing, with
    some sets of learnt weights.

-   @Newell2017PixelsEmbedding use a CNN (ie for images) to construct
    scene graphs from images. The graph itself is constructed from
    additional output channels from the CNN, in which pixels produce
    embeddings, and more similar embeddings are considered to represent
    a relationship between the respective pixels.

-   @Klawonn2018GeneratingConstruction propose a scene graph generator
    that does not require pre-existing grounded captions. Their approach
    is briefly to construct individual relationship statements using an
    attention GAN, then constuct these relationships into one scene
    graph. They do not employ message passing, but instead an LSTM
    network as a constructor.

-   @Qin2017GeneratingPrivacy propose Local Differential Privacy
    Generator (LDPGen) for producing locally private synthetic social
    graphs. Their approach consists of three phases: 1. Initial
    grouping, in which users are partioned into equal sized groups, with
    this grouping scheme communicated to all users. Each user forms a
    degree vector to each of the communities, and adds noise according
    to a privacy budget, following which a new grouping scheme is
    computed with varying group sizes. 2. Grouping refinement, in which
    one or more extra rounds of grouping. 3. Graph generation, in which
    BTER [@Seshadhri2012CommunityGraphs] is used to generate a synthetic
    graph from the degree vectors for these groups.

-   @Bjerrum2017MolecularRNNs address molecule generation with an LSTM
    network with molecules encoded as SMILES. This is an early paper,
    and doesn't use GNNs.

-   @Yoon2023GraphNetworks propose the Computation Graph Transformer
    (CGT) to learn and reproduce real-world graphs in a
    privacy-controlled manner. They approach graph generation as a
    sequence generation problem. Specifically they approach generation
    of "computation graphs" (ie ego networks), as they identify that on
    large graphs these are the most commonly used. They adapt XLNet (ADD
    CITATION) to produce these computation graphs, using flattened
    sequences with graph-position embeddings to encode the graph
    structure. Lots of interesting work here on privacy preservation for
    these graphs.

## GNNs

-   @Brockschmidt2018GenerativeGraphs propose to use graphs to generate
    code. It structures code as graphs, then uses an encoder to obtain
    representations for all nodes. They augment each node with two new
    nodes: one represents inherited information, and the other
    synthesised information. A grammar-driven decoder is used to
    sequentially generate the AST (Abstract Syntax Tree) and the
    corresponding program. They also propose a new dataset.

-   @Kipf2016VariationalData propose the Variational Graph Auto-Encoder
    (VGAE). They optimise on the KL divergence, and test on edge
    prediction, but this is only a 2 page work.

-   @Wang2017GraphGAN:Nets propose GraphGAN, a Generative Adversarial
    Network (GAN) for undirected graphs. This isn't actually intended to
    generate whole graphs as-such, but instead as a representation
    learning model for other downstream tasks.

-   @You2018GraphRNN:Models propose GraphRNN, a recurrent model for
    generating graphs. This is the first paper to propose using MMD, and
    also one of the first autoregressive models. Extends up to 500
    nodes, $O(|V|^2)$, but does not handle node, edge or graph features
    or labels.

-   @Bojchevski2018NetGAN:Walks use biased random walks across an
    initial graph to constuct graphs. The output graph is sampled based
    on how often edges and nodes appear in these random walks. Handles
    directed graphs, but not features, and (arguably) represents more of
    a re-arrangement than sampling a new graph. @Jalilifard2019CanWalks
    propose a new way of initialising NetGAN walks with a set of dense
    vertices.

-   @DeCao2018MolGAN:Graphs propose a GAN to produce molecules, guided
    by an RL objective to produce desired chemical properties. It
    produces close to 100% valid compounds. The RL objective function
    lets them optimise for non-differentiable goals, and use Deep
    Deterministic Policy Gradients (DDPG), an actor-critic model. The
    GAN $G_{\theta}$ learns an approximation of the reward function
    $\hat{R_{\psi}}(G)$, and a reward network is trained to minimise
    between actual reward from a generated graph and this predicted
    reward. The generator then learns to maximise this reward
    $\hat{R_{\psi}}(G)$. Their discriminator and reward network are both
    Relational-GCN .

-   @Simonovsky2018GraphVAE:Autoencoders propose a variational
    auto-encoder for graph structures - node that this is not the same
    as VGAE above. Its very expensive ($O(|V|^4)$) but can handle all
    types of features and in theory perform conditional generation.

-   @Zhou2019Misc-GAN:Graphs propose a multi-scale GAN for graph
    structures Misc-GAN. It generates sections of graphs at different
    levels of granularity, with the model learning to define said
    granularity, then uses a "graph reconstruction model" to re-build
    into a single graph. It scales well ($|V| \sim 10000$) but has
    similar performance to E-R and BA rule-based models.

-   @Liao2019EfficientNetworks propose the Gated Recurrent Attention
    Network (GRAN). It is an auto-regressive model, with best-in-class
    scaling ($O(|V|\log{|V|})$) but does not produce any features or
    have the capacity for conditional generation.

-   @Zhang2019D-VAE:Graphs propose a graph VAE specifically for directed
    acyclic graphs (DAGs), called D-VAE. They additionally propose an
    asynchronous message passing scheme to allow encoding the
    computations on DAGs. This just means that in order for a message to
    be passed to a successor, the pre-decessor must have finished
    computing its own states. They use this D-VAE primarily for
    optimisation and performance prediction of neural networks.

-   @Wang2022DeepGraphs propose a deep-learning encoder-decoder PGD-VAE
    framework to produce periodic graphs. Periodic graphs consist of
    repetitive local structures, such as crystals or polygons.

-   @Guo2021GeneratingAutoencoders predict protein structures using
    several deep architectures. They demonstrate that they show promise
    in this regard.

-   @Du2022InterpretableConstraints propose monotonically-regularised
    graph variational autoencoders. These VGAEs learn to represent
    graphs (molecules) with latent variables, and the relationship
    between these variables and molecular properties using polynomial
    functions. They also derive new objectives which enforce
    monotonicity between these latent variables and molecular
    properties.

-   @Popova2019MolecularRNN:Properties propose MolecularRNN, a model for
    generating graph structures recurrently. This is an adaption of
    @You2018GraphRNN:Models's GraphRNN model for molecules, with
    training according to policy gradient to target molecular
    characteristics.

-   @Knyazev2021ParameterArchitectures propose to use GNNs to generate
    edge weights in neural networks - ie parameter prediction. It scales
    exceptionally well (up to 24m parameters on ResNet-50) as it follows
    the bi-level optimization paradigm, which is common in
    meta-learning. They use a GatedGNN ADD CITATION.

-   @Jin2018JunctionGeneration propose a junction-tree VAE. It first
    generates a tree-structured scaffold over molecular substructures
    (ie motifs) and then uses combines them into a complete molecule
    using a VAE.

-   @Khodayar2019DeepGrids propose a model for generating power grids.
    This model uses a Gated Recurrent Unit (GRU). They generate very
    large grids ($|V| \sim 14000$) with realistic features in a manner
    similar to GRAN [@Liao2019EfficientNetworks]. Node states are
    calculated using a simple matrix multiplicative message passing
    *without learnt features* which should significantly decrease
    computational costs.

-   @Su2019GraphNetwork introduce the Graph Variational Recurrent
    Variational Neural Network (GraphVRNN). It aims to capture the joint
    distributions of graph structures and underlying node attributes.
    They map graphs to sequences first using BFS, then generate those
    sequences auto-regressively.

-   @Goyal2018DynGEM:Graphs propose GraphGen, which takes a
    non-message-passing approach to graph generation. It first converts
    graphs to minimum DFS codes, which also captures label information,
    then uses an LSTM architecture to generate these codes. This is far
    more efficient than GNNs. @Bacciu2021GraphGen-Redux:Generation
    propose GraphGen-Redux, with a novel pre-processing step that
    alleviates assumptions of independence and is significantly lighter.

-   @Bacciu2020Edge-basedNetworks again propose to treat graph
    generation as a sequential problem. Their generation process is
    different, in that they use twin RNNs, the first of which predicts
    an edge start-point, and the second of which perdicts an edge
    end-point.

-   @Podda2020AGeneration propose to produce molecules in "fragments"
    (motifs) instead of atom-by-atom.

    @Ma2018ConstrainedAutoencoders propose a regularisation framework
    for molecular VAEs to ensure specific chemical laws (bound counts
    etc) are followed. They focus on the adjacency matrix, and formulate
    loss terms that penalise if validity constraints are not followed.
    Their main takeaway is that generated graphs are more likely to be
    chemically valid.

-   @Bresson2019AGeneration identify that decoding vector
    representations of molecules is an open problem. They propose a
    two-step decoder: the first generates a bag-of-atoms representation,
    ie $C_3 H_8$, then the second predicts edges from this bag-of-atoms.
    This approach improves their reconstructive scores significantly
    ($76.7\% \rightarrow 90.5\%$).

-   @Gamage2020Multi-MotifGANPrediction generalise NetGAN
    [@Bojchevski2018NetGAN:Walks] to produce graphs of a larger scale
    using random walks, each of which targets a different structural
    motif. The motifs in this work are small ($|C| \leq 3$). As in
    NetGAN they do not aim to reproduce from-scratch networks.

-   @Zhang2019STGGAN:Generation propose STGGAN (Spatio-Temporal Graph
    Generation) to produce graphs with temporal and/or spatial
    attributes for nodes and edges. It uses spatio-temporal random
    walks, with both generator and discriminator using LSTM
    architectures.

-   @Shi2020APrediction propose GraphAF, a flow-based auto-regressive
    model for molecule generation.

-   @Salha2019Gravity-inspiredPrediction propose a gravity-inspired VAE
    for directed link prediction tasks. They show that it can
    effectively reconstruct graphs from node embeddings, and out-perform
    existing AE and VAE architectures on benchmark datasets.

-   @{Maziarka2020Mol-CycleGAN:Optimization.} propose Mol-CycleGAN, a
    cycle-based GAN that takes as input a molecule and as output
    produces an optimised version.

-   @You2018GraphGeneration propose a "graph convolutional policy
    network" GCPN for molecule generation. It is one of the first
    networks optimised through policy gradient to target chemical
    properties.

-   @Yang2019ConditionalNets propose CondGen, a variational-GAN without
    a domain specific focus, but aimed to allow high-quality conditional
    generation. The GAN is used to ensure that graph embeddings from the
    VAE are permutation invariant (ie do not change based on
    node-order).

-   @{Li2018Multi-objectiveModel.} propose a sequential molecule
    generator which samples transitions to iteratively add to or refine
    a molecule.

-   @HowOpenReview review the fundamental limits of message-passing GNN
    kernels, and in particular find that they cannot represent cycles or
    cycle-related features like clustering. Their proposed models, with
    a new GNN kernel, achieves state-of-art performance.

-   @Samanta2020NEVAE propose NEVAE, a VAE for molecular generation that
    also produces 3D coordinates for each atom. Again the decoder is
    trained to optimise certain molecular qualities.

-   @Guo2019DeepCo-evolution propose disentanglement processes for graph
    structures, in particular focusing on molecules, including a novel
    variational objective to disentangle nodes, edges, and their
    co-relations.

-   @Du2022DisentangledModels apply disentangling models to
    spatio-temporal graphs. This includes a variational objective
    function and mutual information thresholding algorithms.

-   @Guo2021DeepNetworks propose SND-VAE, or Spatial-Network VAE, and
    Spatial-MPNN to disentangle and generate spatial networks. This
    again includes a new objective function and an optimisation
    algorithm for that objective.

-   @Flam-Shepherd2020GraphGeneration (BORDERLINE TO INCLUDE) improve
    the Graph-VAE GVAE architecture by including MPNNs in their encoder
    and decoder architectures.

-   @Honda2019GraphGeneration propose Graph Residual Flows (GRF) for
    molecule generation, and demonstrate that these flows are
    invertible. Their model achieves comparable performance while using
    fewer trainable parameters.

-   @Joon-WieTann2020SHADOWCAST:Generation use Conditional-GAN to
    produce graphs that, using a Markov model, takes an observed graph
    and Markov parameters to control graph generation.

-   @Kawai2019ScalableMechanism propose GRAM, which they argue is a more
    efficient graph generation model, thanks to its novel attention
    mechanisms. This is an auto-regressive model. Their benchmarks are
    the grid, lobster, community and Barbasi-Albert datasets.

-   @Madhawa2019GraphNVP:Graphs propose GraphNVP, the first invertible,
    normalising flow-based molecular graph generation model. Generation
    is first of an adjacency tensor, then node attributes. This includes
    two novel reversible flows. They evaluate their models using
    chemical datasets.

-   @Yang2020LearnNets propose the Time Series Conditioned Graph
    Generation GAN (TSGG-GAN) to generate graphs with conditioning on
    time series, for example gene interactions conditioned on a certain
    disease over time.

-   @Ahn2020GuidingExploration propose Genetic Expert-Guided Learning
    (GEGL) to produce high-reward molecules. This is heavily based on RL
    techniques, with one actor (neural apprentice) creating molecules,
    and another (genetic expert) creating molecules based on those
    already produced. The first actor then optimises itself according to
    the union of these two sets of produced graphs. Molecules themselves
    are generated using SMILES representations.

-   @Tran2022DeepNC:Completion propose DeepNC, a graph completion model,
    designed to add missing sections back into graphs. They achieve
    excellent runtimes, almost linear with number of nodes in a graph.
    The generative process itself is autoregressive.

-   @Ingraham2019GenerativeDesign identify the complex issue of the
    coupling between protein sequence and 3D structure. This is often
    refered to as the "inverse protein folding" problem. They address
    this problem through conditioning the protein sequences on the graph
    specification of the target structure.

-   @W2022IterativeCo-design propose Syntax-Directed-VAE or SD-VAE. They
    use this syntax to constrain the output of the decoder, ensuring
    semantic validity and reasonability.

-   @Dai2020ScalableGraphs propose BiGG, which aims at producing sparse
    graphs. As such they achieve $O(\log(|V|))$ with their
    auto-regressive model. Their model boosts scalability using an
    R-MAT-like [@Chakrabarti2004R-MAT:Mining] division of the adjacency
    matrix, producing each row at a time. They generate un-attributed
    graphs up to 20k nodes, but on very sparse graphs.

-   @Ling2021DeepNetworks propose Heterogenous Graph Generation (HGEN)
    to produce heterogenous graphs. It uses a walk generator that
    hierarchicaly generates meta-paths and their path instances. They
    also include a heterogenous graph assembler that combines these
    walks into a single graph.

-   @Luo2021GraphDF:Generation propose GraphDF (Discrete Flow) for
    molecule generation. It uses "invertible modulo shift
    transformations" to map discrete latent variables to graph
    structures. It is another sequential (auto-regressive) model, with
    labelled nodes and edges added in that order. It supports node and
    edge labels and conditional generation through an external
    classifier or regressor's gradients.

-   @Polsterl2021AdversarialGeneration propose ALMGIG, a GAN with
    cycle-consistency loss to enforce reconstruction. They extend the
    Graph Isomorphism Network (ADD CITATION) to multi-graphs to account
    for n-covalent bonds. They also propose to use the 1-Wasserstein
    distance between chemical properties to quantify performance.

-   @{Griffiths2019ConstrainedAutoencoders.} use constrained Bayesian
    optimisation to improve the generative capacities of VAEs. This
    allows their VAEs to perform better when the latent space is sampled
    further from the training data.

-   @Gomez-Bombarelli2018AutomaticMolecules explore a method to convert
    discrete molecular representations to-and-from continuous vectors.
    They produce an encoder-decoder architecture (similar to VAE), and a
    predictor for chemical properties trained on the continuous vectors.

-   @Du2021DeepGeneration propose a VAE for molecule generation with a
    focus on disentangled representation learning. They argue that their
    approach allows inductive links between learned latent factors and
    molecular properties.

-   @Guo2020PropertyDependence (NOT GRAPHS) propose the Property
    Controllable VAE (PCVAE), where a Bayesian model is used to
    inductively bias the latent representation of the VAE using explicit
    data properties. This is achieved through novel group-wise and
    property-wise disentanglement. This means that each latent variable
    corresponds to a data property, as invertible mutual dependence is
    enforced between them.

-   @Vignac2021Top-N:Exchangeability propose Top-N creation, a
    differentiable generation mechanism that uses a latent vector to
    select the most relevant points from a trainable reference set.
    (COME BACK TO, QUITE TECHNICAL)

-   @Niu2020PermutationModeling propose a score-based model to model the
    gradient of the data distribution at the input graph (the score
    function). This produces a permutation equivariant model of
    gradients implicitly defines a permutation invariant distribution
    over graphs. They train this network through score-matching and
    sample from it using annealed Langevin dynamics.

-   @Liu2021GraphEBM:Models propose GraphEBM (Energy Based Models) for
    molecule generation, which use a permutation-invariant formulation
    of their energy function to in turn make molecular generation a
    permutation-invariant problem. They apply Langevin dynamics to train
    the energy function by maximising likelihood and generate samples
    with low energies. They use additional terms in their energy
    function to target molecular properties in conditional generation.

-   @Liu2018ConstrainedDesign propose a constrained graph-VAEs for
    molecule design. Both encoder and decoder are graph-structured,
    which at publication was fairly novel. Their decoder assumes a
    sequential ordering of graph extension steps. Their principle aim is
    that sets of generated graphs more closely conform to distributions
    in the training data.

-   @Zang2020MoFlow:Graphs propose MoFlow, a flow-based generative model
    to learn invertible mappings between molecular graphs and their
    latent representations. It first generates bonds through Glow (ADD
    CITATION), then atoms given these bonds through a novel graph
    conditional flow, then assembles the result through a post-hoc
    validity correction.

-   @Li2020DirichletAutoencoder propose the Dirichlet Graph VAE (DGVAE),
    a VAE with graph cluster membership as latent factors. This combines
    VAE based generation with balanced graph cuts, providing insight
    into the internal mechanisms of VAEs on graph structures. They also
    propose a novel GNN variant Heatts which encodes a graph into
    cluster memberships, which they argue has better low-pass
    characteristics than GCNs.

-   @Grover2019Graphite:Graphs propose Graphite, an unsupervised
    algorithm for learning representations of nodes in large graphs
    using latent variable generative models. Specifically they use a VAE
    with an iterative graph refinement strategy inspired by low-rank
    approximations for decoding. They don't specifically use this to
    generate graphs.

-   @Zhang2019AMRTransduction propose an attention-based AMR (Abstract
    Meaning Representation) parsing model as a sequence-to-graph
    problem. This is essentially a graph translation task.

-   @Anand2018GenerativeStructures apply GANs to generating protein
    structures. Proteins are encoded in terms of pairwise distances
    between alpha-carbons which reduces the need to learn
    translational/rotational symmetries. They evaluate by completing
    corrupted ground-truth protein structures.

-   @Fan2019LabeledNetworks propose LGGAN (Labelled Graph GAN) for graph
    structures with node labels. They evaluate on citation and protein
    networks up to $|V| \sim 200$.

-   @Sun2019GraphGeneration a graph-to-graph translation work, focussing
    on topology. They embed the topology of the source into node-states
    through exerting a topology constraint, resulting in a Topology-Flow
    encoder. Decoding is through a conditioned graph generation model
    with two modes, resulting in their Edge-Bernoulli decoder and their
    Edge-Connect decoder. They evaluate on small graphs ($|V| < 100$).

-   @DArcy2019DeepGeneration aim to generate directed acyclic graphs
    using reinforcement learning techniques. It adds nodes
    auto-regressively, and of course handles directed edges, but they
    describe the original network as similar to graphSAGE (add
    citation).

-   @Kearnes2019DecodingLearning propose RL-VAE (Reinforcement Learning)
    for decoding molecules from latent embeddings. They argue that not
    having both an encoder and decoder allows lower training and
    evaluation complexities.

-   @Rivas2019DiPol-GAN:Pooling introduce DiPol-GAN for molecule
    generation. DiPol here is Differentiable Pooling, not Dipole. They
    constrain the learned latent representation with an RL objective to
    shift generation towards a target property.

-   @Liu2019GraphFlows introduce graph normalizing flows, with a
    reversible GNN model for prediction and generation. They demonstrate
    that on supervised tasks their normalizing flows are more efficient
    than MPNNs. They then combine these flows with a novel auto-encoder
    to create a generative model of graph structures, which is
    permutation-invariant.

-   @Kaluza2018ATranslation develop a DAG-to-DAG translation model.

-   @Chen2018Sequence-to-Action:Parsing approach the sequence-to-action
    semantic problem with graphs. Their model first uses a semantic
    graph to represent the meaning of a sentence, then an RNN translates
    that graph into actions. They are competitive on relevant
    benchmarks.

-   @Allamanis2018LearningGraphs propose to use graphs to represent
    program code. This results in very large graphs, and they do not
    attempt generation, but they do demonstrate that graphs are a good
    representation for code.

-   @Jin2020HierarchicalMotifs propose a hierarchical molecule
    generator. They extend the graph-motif concept from small molecular
    blocks to larger, more flexible sections. Their model is an
    encoder-decoder architecture, with the encoder producing a
    multi-resolution representation for each molecule, from atoms to
    connected motifs. The decoder adds motifs sequentially, alternating
    adding motifs and connecting said motif to the existing graph.

-   @Karimi2020Network-principledSets generate drug combinations as sets
    of graphs using a hierarchical VAE trained through RL. Their VAE
    jointly embeds gene-gene, gene-disease and disease-disease graphs.
    The drug combination design problem is then generating sets of
    graphs. Specifically, along with chemical validity rewards, they
    propose a new RL policy term "sliced Wasserstein", which targets
    chemically diverse molecules with distributions like those of real
    drugs.

-   @Khemchandani2020DeepGraphMolGenApproach develop DeepGraphMolGen, a
    multi-objective strategy for generating molecules using GCNs and RL
    policies. Interaction binding models are learnt using GCNs, using a
    loss function with chemically targetted terms, including synthetic
    accessibility. Multi-objective reward functions allow them to design
    drugs that can target one receptor but avoid another.

-   @Trivedi2020GraphOpt:Formation propose GraphOpt to optimise graph
    formation models. It jointly learns an implicit model of graph
    structural formation and an underlying optimisation mechanism
    through a latent objective function. Graph formation is posed as a
    sequential process, solved through a maximum entropy inverse RL
    algorithm. It also employs a continuous latent action space to aid
    scalability.

-   @Stier2021DeepGG:Generator propose DeepGG, a framework for learning
    generative models of graphs based on deep state machines. State
    transition decisions they use a set of graph and node embedding
    techniques as the memory state. Their model learns the distribution
    of random graph generators, with evaluation of how well each
    property is learnt and which distributions are preserved.

-   @Stoehr2019DisentanglingGraphs propose to use Beta-VAE to approach
    graph decoding as the inverse of graph encoding, as they argue that
    this leads to interpretable parameters arising naturally in the
    latent space. They measure the degree of disentanglement with the
    Mutual Information Gap (MIG), and evaluate on ER graphs, showing a
    near one-to-one mapping between latent variables and the parameters
    $|V|, P$.

-   @Yang2018GraphGeneration propose Graph R-CNN for scene graph
    generation. Their model contains a Relation Proposal Network (RePN)
    to deal with relations between objects, with an attention-GCN to
    capture contextual information between objects and relations.

-   @Qi2019AttentiveGraphs propose a 2-stage method for scene graph
    generation. In the first stage, a semantic transformational module
    embeds visual and linguistic features into a common semantic space.
    The second module, a graph self-attention model, embeds a joint
    graph representation. Finally scene graphs are produced by a
    relation inference module to recognise entities and corresponding
    relations.

-   @Khademi2020DeepGeneration propose Deep Generative Probabilistic
    Graph Neural Networks (DG-PGNN), a scene graph generation algorithm.
    Taking an image with region-grounded captions, DG-PGNN constructs a
    PGN, like a scene graph with uncertainty. Each node is a CNN
    embedding, and it defines a Probability Mass Function (PMF) for
    node-types and edge-types. DG-PGNN formulates graph construction as
    a Q-learning problem, adding nodes sequentially, with updated node
    states after each addition.

-   @Li2018FactorizableGeneration handle scene generation through
    partitioning into subgraphs, with each subgraph containing a subset
    of objects and their relations. This required their proposed
    Spatial-Weighted Message Passing (SMP) structure and
    Spatial-sensitive Relation Inference (SRI) to facilitate
    relationship recognition.

-   @Li2018LearningGraphs use GNNs to express probabilistic dependencies
    among a graph's nodes and edges. They show that their models can
    generate quality samples over synthetic datasets and molecular
    graphs. This is an early GNN paper, and is often cited by other,
    newer works.

-   @Zhang2019Circuit-GNN:Design propose Circuit-GNN, a model for
    simulating and optimising circuits. The learnt process is the EM
    properties of distributed circuits, and by leveraging the
    differentiability of NNs, it is able to generate new circuits given
    some EM properties. Against a commercial simulator their model is up
    to four orders of magnitude quicker, and is able to generate complex
    circuit designs that previously would require a complex expert
    design process.

-   @Kan2022FBNETGEN:Generation propose FBNETGEN, a GNN model for
    generating functional brain networks. They formulate predictions for
    prominent regions of interest, brain network generation and clinical
    predictions in an end-to-end model. Their key contribution is their
    graph generator, which translates raw time-series features into
    task-oriented brain networks.

-   @Palowitch2022GraphWorld:GNNs develop GraphWorld, a methodology and
    system for benchmarking GNN models on an arbitrarily-large set of
    synthetic graphs for any conceivable GNN task. It includes, of
    course, graph generators to create these datasets. They argue that
    their system is better able to explore the whole space of possible
    graphs. Each graph set is sampled from the paramaterised probability
    distribution $P(\pi_1, \pi_2, ...)$ on $D = G \times F \times L$
    where $G$ is a collection of graphs, $F$ a collection of features,
    and $L$ a collection of labels. They seem to primarily use the
    Degree-Corrected SBM (DC-SBM, ADD CITATION) generator.

-   @Hu2020GPT-GNN:Networks propose GPT-GNN, a generative pre-training
    framework for GNNs. The pre-training task here is attributed graph
    generation, with the likelihood of graph generation factorised into
    attribute and edge generation. They demonstrate SOTA performance.

-   @Simm2020AGeometry address the problem of molecular geometry - ie
    given a graph structure, how is the molecule actually arranged in
    3D? They utilise a CVAE (ADD CITATION), and encode atomic distances
    instead of cartesian atom coordinates. They also show how their
    model can be used in an importance sampling scheme to compute
    molecular properties.

-   @Xu2018Graph2Seq:Networks propose Graph2Seq, a model for
    graph-to-sequence learning. This is essentially an extension of
    Seq2Seq (ADD CITATION) to graph inputs. They use message passing
    with an LSTM architecture. Not a strict graph generation model.

-   @Chen2021OrderGeneration identify that for sequential models node
    ordering creates issues in maximum likelihood estimation due to the
    large permutation space. As such they present an expression for the
    likelihood of a graph generative model, and link it to graph
    automorphism. They then develop a variational inference model for
    fitting graph-generative models that maximises the variational bound
    of the log-likelihood. This allows training with node orderings from
    the approximate posterior instead of ad-hoc orderings.

-   @Diamant2023ImprovingBandwidth aim to improve graph generation
    models (both auto-regressive and one-shot) by restricting the
    possible bandwidth of generated graphs. They find that, using
    existing models, this restriction results in better scalability,
    generation quality and reconstruction accuracy.

-   @Ma2023GeneratedDetection propose a set of classifiers for detecting
    generated graphs. They propose four classification scenarios, with
    each scenario switching between seen and unseen datasets and
    generators during testing, to progressively test classifiers and
    mirror real-world conditions. They demonstrate good performance,
    with different classifiers carrying different advantages.

-   @Klepper2022RelatingModels prove that the solution space induced by
    graph auto-encoders is a subset of the solution space of a linear
    map. They then argue that this represents a useful inductive bias -
    a reduced space actually improving results. They identify the node
    features as a powerful inductive bias. Introducing a corresponding
    bias in a linear model they demonstrate that a linear encoder can
    out-perform a non-linear encoder.

## Diffusion

-   @Lee2023ExploringGeneration propose a score-based diffusion model
    Molecular Out-Of-Distribution Diffusion (MOOD). It incorporates
    out-of-distribution control in the generative SDE with a single
    parameter. Conditional generation is performed using the gradients
    from a property predictor.

-   @Vignac2023DiGress:Generation propose DiGress, which employs
    discrete denoising diffusion for graph generation. The noise
    function here is defined through label marginals, with the
    presence/absence of a node treated as labels. It is a one-shot
    method, producing whole graphs at a time, and supports labels for
    nodes, edges and graphs. Conditional generation is through the
    gradient of a feature predictor.

-   @Luo2021DiffusionGeneration propose a diffusion model for point
    cloud data. Their model is highly thermo-inspired, modelling points
    as particles in a thermodynamic system in contact with a heat bath.
    The reverse diffusion process is then returning to their original
    coordinates. This is conducted through a Markov chain.

-   @Hoogeboom2022Equivariant3D propose a diffusion model for molecule
    generation in 3D, which is equivariant to Euclidean transforms. This
    Equivariant Diffusion Model (EDM) uses an equivariant network that
    operates both on continuous coordinates and discrete atom types.

-   @Jo2022Score-basedEquations propose a score-based diffusion model
    that models the joint distribution of nodes and edges through a
    system of SDEs. This includes a set of novel score matching
    objectives aimed at the joint-log-density between nodes and edges,
    as well as a new SDE solver. It achieves very competetive results on
    benchmark datasets.

-   @Huang2022GraphGDP:Generation propose GraphGDP (Generative Diffusion
    Processes) for permutation invariant graph generation. For their
    inverse noise process they propose a position-enhanced graph score
    network, aimed at capturing evolving structure and position through
    the inverse noise process for permutation equivariant score
    estimation. They show that GraphGDP can generate high-quality graphs
    in only 24 function evaluations.

-   @Huang2022ConditionalGeneration propose a conditional diffusion
    model for molecule generation. Their forward noise process operates
    on both graph structure and inherent features, and they derive
    discrete graph structures as the condition for the reverse process.
    They utilise ODE solvers for fast sampling, and demonstrate
    high-quality molecular graphs in a small number of evaluation steps.

-   @Luo2022FastDiffusion propose Graph Spectral Diffusion Model (GSDM),
    wherein noise is applied on the graph spectrum, instead of on the
    actual graph structure. This, they argue, results in models better
    able to learn graph structural features. Their results beat-out many
    existing GNN models.

-   @Chen2022NVDiff:Vectors proposes NVDiff, which first samples node
    vectors through diffusion. The node vectors in question (DOUBLE
    CHECK) are from the latent space of VGAE, and the authors apply
    another model to decode these node vectors back into a graph
    structure. This reduces the $O(N^2)$ constraint of 1-shot graph
    generators, allowing NVDiff to scale better than other GNN models.
    They evaluate up to $|V| \leq 400$ nodes.

-   @Ye2022FirstData propose First Hitting Diffusion Models (FHDM),
    where the inverse noise process terminates at a random first hitting
    time. Their aim is that a process can terminate early and still
    produce high(er) quality graphs, while reducing sampling time. They
    observe considerable improvement over SOTA in quality and speed.
    Their hitting conditions are proposed as: pre-fix time, sphere hit,
    boolean (dimensional) hit, categorical hit.

-   @Yang2022Diffusion-BasedPre-Training approach scene-graph to image
    translation using a diffusion model. Specifically they propose to
    learn scene-graph embeddings by directly optimising their alignment
    with images. They have two embedding components; one learns to
    produce embeddings by re-constructing randomly masked image regions,
    and the other trains to discriminate between compliant and
    non-compliant images according to the scene graph. These embeddings
    are then used to train a diffusion model to produce images from
    scene graphs, which can be tweaked by altering the prior scene
    graph.

-   @Jo2023GraphMixture propose to model, through the inverse process,
    the topology of graphs by predicting the destination of said inverse
    process. To achieve this they model the generative process as a
    mixture of diffusion processes conditioned on the endpoint in the
    data distribution, which drives the process towards the probable
    destination. This includes new training objectives for learning to
    predict that destination, and discuss how this allows their model to
    exploit the inductive bias of the data. They evaluate on general
    graphs and the 2d/3d molecular domain, including discrete and
    continuous features. The datasets they use are in the
    hundreds-of-nodes range, comparable to DiGress
    [@Vignac2023DiGress:Generation].

-   @Yan2023SwinGNN:Generation propose SwinGNN, which makes use of
    shifted window self-attention layers, inspired by SwinTransformers
    (ADD CITATION). Their main contribution is that their model is not
    invariant (unlike other GNN models), and so avoids issues with the
    number of modes in their target distributions and the number of
    components in their denoising Gaussians (DOES THIS APPLY TO DISCRETE
    DIFFUSION?). Like other diffusion models they state that their model
    is SOTA on their benchmarks.

-   @Wen2023HyperbolicGeneration propose the Hyperbolic Graph Diffusion
    Model (HGDM) for molecule generation, which employs a hyperbolic VAE
    to learn the hidden representation of nodes, then a score-based
    hyperbolic GNN to learn said distribution in hyperbolic space.

-   @Zhang2023LayoutDiffusion:Models use discrete diffusion models to
    produce LayoutDiffusion, a model for graphic layout. They develop a
    noise process based on the legality, coordinate proximity and type
    disruption in the noisy layout, which considers the continuous and
    discrete features present.

-   @Limnios2023SaGess:Generation propose SaGess for large generation.
    Using DiGress [@Vignac2023DiGress:Generation] as a base for
    generating sub-graph components of a larger graph. These components
    have node ids from the original graph as features (ie labels). This
    means that the new large graph can be sampled piece-at-a-time, with
    each component having node ids as labels, and unique edges added
    until the original graph's edge count is reached.
    @Limnios2023SaGess:Generation evaluate on graphs up to 2700 nodes
    with Cora.

-   @Tseng2023GraphGUIDE:diffusion propose GraphGUIDE, a framework for
    more interpretable and controllable conditional generation from
    diffusion models. Here edges in the graph are flipped or set at each
    timestep, through bit-kernels, which the authors argue results in a
    more interpretable and controllable generative process.

-   @Rong2023Complexity-awareModel use a graph diffusion model to model
    the flow of people within cities (Origin-Destination graphs, OD). To
    reduce computational complexity they treat the generation of edge
    weights as a secondary step.

-   @Kong2023AutoregressiveGeneration propose an auto-regressive
    diffusion model for graph generation, which they argue alleviates
    computational costs and allows the inclusion of constraints during
    sampling. Their forward noise process is a learnt node-absorption,
    data dependent, that learns an absorption order from topology. The
    inverse process is then node adsorption, ie sequential node
    addition. This includes node labels. They show that these two
    networks can be jointly trained by optimising the lower bound of
    data likelihood. As in other papers they claim SOTA performance on
    benchmarks, and fast generation speeds.

-   @Jo2022Score-basedEquations propose a fairly generic graph diffusion
    approach, modelling the joint distribution between nodes and edges,
    and introducing a solver for the SDEs of their inverse noise
    process.

-   @Xu2023GeometricGeneration propose GeoLDM (Geometric Latent
    Diffusion Model), a latent diffusion model for 3D molecule
    generation. As in other latent DMs their noise (and inverse-noise)
    process is applied on the latent space of molecules from
    autoencoders. Their main contribution is modelling the 3D molecular
    geometries critical roto-translational equivariance by building a
    point-structured latent space with both invariant scalars and
    equivariant tensors.

-   @KonstantinHaefeliETHZurich2022DiffusionSpaces demonstrate that
    graph diffusion models benefit from discrete state-spaces - a point
    implemented in most of the graph diffusion works following this
    paper. Discrete noise means that at every noise step the graph is
    still discrete, and as such more stable under GNNs. They demonstrate
    a reduction in sampling steps from 1000 down to 32.

-   @Tseng2023ComplexDiffusion explore how different discrete diffusion
    kernels, which converge to different prior distributions, affect
    generative performance for GNNs. They show that the quality of
    generated graphs is indeed sensitive to the prior used, but that the
    best prior to use cannot be predicted from obvious statistics.

-   @Xu2023GeometricGeneration propose GeoDiff for molecular
    conformation prediction. The challenge, they note, is in producing
    likelihoods of conformations that are roto-translational invariant.
    They design a Markov chain with an evolving Markov kernel that
    induces this by default, and propose further building blocks that
    preserve the desireable equivariance property.

-   @Yi2023GraphFolding propose to use graph denoising diffusion for
    inverse protein folding. They identify that there are many solutions
    to a given protein folding problem, and that existing models
    struggle to encapsulate this diverse range of plausible options.
    Their proposed denoising diffusion model is guided by a protein
    backbone to produce amino acid residue types. The model infers the
    joint distribution of amino acids conditioned on nodes'
    physiochemical properties and local environment. Their forward
    process uses amino acid replacement matrices, encoding alongside
    these acids biologically-meaningful prior knowledge about them and
    their neighbours.

-   @Hwang2023EfficientGraphs propose a similar method to SaGess
    [@Limnios2023SaGess:Generation] and HiGGs
    [@Davies2023HierarchicalGeneration], using DiGress
    [@Vignac2023DiGress:Generation] to produce graph motifs. Unlike
    SaGess or HiGGs, they only use one pass of DiGress, instead
    compressing training graphs using common few-node motifs. As such
    their achieved scales are lower, but they achieve superior
    performance to vanilla DiGress on their benchmark datasets.

-   @Davies2023HierarchicalGeneration take a similar approach to SaGess
    [@Limnios2023SaGess:Generation] in using a conditional GNN model
    (here DiGress, [@Vignac2023DiGress:Generation]) to produce subgraphs
    with their HiGGs framework, which are then assembled into a larger
    graph. Here three separate models are used. The first produces a
    community-community meta-graph, ie a graph of how communities
    connect to each other, with node labels indicating the majority
    class of said community. The second produces community subgraphs for
    the nodes in this community-community graph. The last predicts edges
    between these sampled communities for each edge in the
    community-community graph. The second and third steps are
    paralellisable, though the authors do not do this in their paper,
    and HiGGs is implemented up to a $|V| \sim 22k$ node graph.

-   @Yang2023DirectionalLearning propose to use diffusion models for
    unsupervised graph representation learning. They propose directional
    diffusion models, arguing that adding isotropic noise to
    an-isotropic graphs may happen too quickly. These directional
    diffusion models incorporate data-dependent, anisotropic,
    directional noises in the forward diffusion process. They
    demonstrate that this representation learning scheme achieves SOTA
    performance.

-   @Huang2023MDM:Generation develop a diffusion model for 3D molecule
    generation. They attempt to address the twin issues of degrading
    performance as molecule scale increases and lack of diversity in
    those generated samples. They propose to use twin equivariant
    encoders to encode forces of different strengths. They note that
    existing models essentially shift elements in geometry along the
    gradient of data density, and that such a process lacks exploration
    of the intermediate steps of the Langevin dynamics. As such they
    introduce a distributional controlling variable in each
    noise/inverse noise step to enforce exploration and sample
    diversity.

-   @Wen2023DiffSTG:Models propose a diffusion model DiffSTG for
    spatio-temporal graphs (STG). As part of this work they propose
    UGnet, a U-Net-like architecture for graphs. They don't strictly
    generate graphs, instead predicting changes in an STG.

-   @Zhai2023CommonScenes:Graphs propose CommonSense, a generative model
    for converting scene graphs into controllable 3D scenes, with
    common-sense semantic realism. Their model consists of two branches.
    The first is a VAE predicting overall scene layout. The second is a
    latent diffusion model generating shapes. This work also includes a
    dataset SG-FRONT, an augmentation of 3D-FRONT to include scene
    graphs.

-   @Anand2022ProteinModels address protein design (both 3D and chemical
    properties). Their model produces both 3D structure and sequence for
    the protein backbone. Its generation is conditioned on a compact
    specification of protein topology and produces full-backbone
    configurations as well as sequence and side-chain predictions. At
    the time of publication it was able to scale better than other
    models.

-   @Bao2022EquivariantDesign address inverse molecular design, using
    Equivariant Energy-Guided Stochastic Differential Equations
    (EEGSDE), an energy guided diffusion model. Under their designed
    energy functions EEGSDE out-performs other baselines on QM9, with
    targetting on quantum properties and molecular structures. Linear
    combinations of energy functions allow targetting of multiple
    chemical properties simultaneously.

-   @Lin2022DiffBP:Binding address target protein binding with DiffBP.
    Their work introduces a diffusion model for molecular 3D structures
    with target proteins as contextual constraints. Given a 3D binding
    site, their model produces both element types and 3D coordinates for
    a whole molecule, with an equivariant network.

-   @Morehead2023Geometry-CompleteOptimization propose the Geometry
    Complete Diffusion Model (GCDM) for 3D molecular generation. Their
    aim is to address identified issues with other models using
    molecule-agnostic and non-geometric GNNs as architectures.
