# Concepts

This page explains the ideas behind the library, starting from NEAT. It follows
Miikkulainen et al., *Evolving Deep Neural Networks*
([arXiv:1703.00548](https://arxiv.org/abs/1703.00548)), and points out where this
implementation had to fill gaps or chose differently.

## The two-level idea

Designing a deep network involves choices that gradient descent can't make: how
many layers, how they connect, what the learning rate is. Evolution is good at those
discrete choices but poor at setting millions of weights. So the work is split:

```text
Outer loop (evolution):   searches architectures and hyperparameters
Inner loop (your code):   trains each candidate with gradient descent
                          and reports how well it did (its fitness)
```

Fitness is measured after a short training budget, so evolution tends to favour
networks that *learn quickly*. The paper notes this bias too. It is often useful,
but a network that learns slowly and ends up better can lose.

## NEAT in brief

NEAT (NeuroEvolution of Augmenting Topologies) evolves graphs. Four ideas carry over:

- **Start minimal.** Initial genomes are as small as possible, and structure grows
  through mutation. Complexity has to earn its place.
- **Historical markings.** Every new connection gets an *innovation number*. Two
  genomes can then be lined up gene by gene for crossover without comparing graphs.
- **Speciation.** Genomes are grouped by similarity and compete mainly inside their
  group. A new structure usually scores badly at first, and this gives it time to
  improve.
- **Crossover by gene.** Genes that match come from either parent. Genes that don't
  match come from the fitter parent.

## DeepNEAT

DeepNEAT (paper section 3.1) keeps that process but changes what a genome means:

| | NEAT | DeepNEAT |
|---|---|---|
| A node is | a neuron | a **layer**, with a table of hyperparameters |
| An edge is | a weighted connection | "this layer feeds that layer" (no weight) |
| Weights come from | evolution | gradient descent, inside the evaluator |
| Also evolved | nothing else | global hyperparameters such as the learning rate |

Real-valued hyperparameters mutate with Gaussian noise and binary ones flip. When a
layer has several parents, a **merge** combines them by concatenation or sum, after
reducing mismatched inputs to the smallest size.

In this library a DeepNEAT individual is one `Genome` of kind `DeepNEAT`, and the
`DeepNEAT` class runs the loop.

## CoDeepNEAT

DeepNEAT networks tend to come out irregular, whereas successful hand-designed
networks repeat a building block many times. CoDeepNEAT (section 3.2) evolves the
blocks and the wiring separately, in two populations:

```text
  Blueprint                 Module species              Assembled network

    (input)              species A     species B            (input)
       |                  [conv]        [conv]                 |
    node -> species A       |          /      \             [conv]   <- module from A
       |                  [conv]    [conv]  [conv]             |
    node -> species B                  \      /             [conv]
       |                                [sum]                  |
    node -> species A                                       [conv]   <- module from B
       |                                                    /    \
    (output)                                            [conv]  [conv]
                                                            \    /
                                                             [sum]
                                                               |
                                                            [conv]   <- module from A again
                                                               |
                                                            [conv]
                                                               |
                                                            (head)
```

- A **module** is a small graph of layers.
- A **blueprint** is a graph whose nodes point to module *species*, not to individual
  modules. It also carries the global hyperparameters.
- To **assemble** a network, each blueprint node is replaced by a module drawn at
  random from its species. If several nodes point to the same species, they all use
  the *same* module, which is what produces repetition.
- The assembled network's fitness goes back to the blueprint and to each module as
  the **average** over all networks they took part in.

Because a small change to one module shows up everywhere it is used, small mutations
cause large structural changes. This lets CoDeepNEAT reach deep, repetitive
architectures that DeepNEAT would rarely find.

## One generation, step by step

`CoDeepNEAT::step` does the following:

1. **Assemble.** Shuffle the blueprints and take them round-robin until
   `networks_per_generation` networks exist. For each, draw one module per
   referenced species. With the paper's 25 blueprints and 100 networks, each
   blueprint appears four times with different module combinations.
2. **Evaluate.** Build each network with fresh random weights and call your
   evaluator, on `num_threads` threads.
3. **Attribute.** Each blueprint's and module's fitness becomes the mean of its
   networks' fitness. A module counts once per network, even if it appears several
   times in that network.
4. **Record.** Keep the genotype of the best single network seen so far.
5. **Reproduce modules.** Cull stagnant species, allocate offspring to species by
   mean fitness, copy elites, breed the rest, and speciate the result.
6. **Reproduce blueprints**, the same way, then re-target any pointer to a module
   species that no longer exists.

Reproduction within a population works like this:

- Each species' mean fitness is normalised across species, and offspring are
  allocated in proportion, with at least `min_species_size` each.
- Species whose best fitness hasn't improved for `stagnation_limit` generations are
  removed, except the top `species_elitism` species.
- The best `elitism` members of a species are copied unchanged.
- Other children come from the top `survival_threshold` fraction, either by crossover
  (with probability `crossover_prob`) or by cloning, and are then mutated.

The compatibility threshold moves after each speciation, aiming for `target_species`. It
starts high relative to the initial differences, which is why the first generations show
few species: all genomes start with identical structure and differ only in hyperparameters.
The target is soft, because the threshold affects only how new species form, and an existing
species lasts until it is culled for stagnation.

## Mutations

| Mutation | Applies to | Effect |
|---|---|---|
| Add node | all | Splits a connection `a -> b` into `a -> new -> b` and disables the original |
| Add connection | all | Adds an edge that keeps the graph acyclic, or re-enables one |
| Toggle connection | all | Enables or disables an edge, never cutting the last input-to-output path |
| Node hyperparameters | all | Mutates entries of a node's table |
| Output merge | all | Flips the merge at the output between concat and sum |
| Add cell skip | modules, DeepNEAT | Adds a memory-cell connection between two LSTM nodes |
| Toggle cell skip | modules, DeepNEAT | Enables or disables one |
| Species pointer | blueprints | Points a node at a different module species |
| Globals | blueprints, DeepNEAT | Mutates the global hyperparameter table |

Rates are listed in [Configuration](configuration.md).

## Recurrent networks and LSTM evolution

Section 4 of the paper extends the search to LSTM connectivity with two mutations:
enabling or disabling connections between LSTM layers, and adding or removing skip
connections between their memory cells. Figure 3 shows the best result, a *feedback*
connection from a later cell to an earlier one.

The first is the ordinary toggle mutation. The second is a separate gene type,
`Genome::cell_skips`. When memory cell *s* connects to memory cell *d*:

```text
c_d(t) = f * c_d(t-1) + i * g  +  sigmoid(W [x(t), h_d(t-1)] + b) * P c_s
```

`P` is a learned projection, present only when the two layers differ in size. The
value of `c_s` depends on execution order. If *s* runs before *d*, it is the current
step's cell. If *s* runs after *d*, it is the previous step's, which gives the feedback
connection in the figure. Skips exist only between LSTM nodes in the same module, or
in the same DeepNEAT genome.

## How this differs from the paper

The paper leaves many details open and evaluated on GPUs with supervised tasks. These
are the choices made here.

| Topic | Paper | This library |
|---|---|---|
| Task | Supervised (images, language, captions) | Any evaluator; RL tools included |
| Modules never drawn in a generation | Not specified | Elites keep their previous fitness; others take their species' mean, or the population mean if no member was evaluated |
| Innovation numbers | Standard NEAT, per generation | One table per population for the whole run, so a structure that reappears aligns with earlier ones |
| Species assignment | "A similarity metric" | Compatibility distance over genes, hyperparameters, species pointers and globals; threshold adapts toward `target_species`, which is a soft target |
| Repeated modules | The same module is used | Same structure, **separate weights** per position |
| Global hyperparameters | Separate chromosome | Stored in the blueprint (`Genome::globals`) |
| Skip gate | Shown in Fig. 3, formula not given | Formula above; gate bias starts at -1 so a new skip begins mostly closed |
| Merge of mismatched sum inputs | Fully connected bottleneck (captioning) | Learned dense projection for flat inputs, 1x1 convolution for spatial ones |
| Mixed spatial and flat merge | Not specified | Everything is flattened |
| Multiple output heads | Not covered | `IOSpec::outputs`, used for actor and critic |
| Weight inheritance | None; trained from scratch | Same. Copy weights yourself if you want a Lamarckian variant ([how](extending.md#weight-inheritance)) |

## Further reading

- [Architecture](architecture.md) maps these ideas to the code.
- [Configuration](configuration.md) lists the settings for everything above.
