# ADRIANNA
Augmented Development Requirements In Adapting Neural Network Architectures

ADRIANNA is a testbed for training neural networks live, while a simulation runs, instead of ahead of time. Each project is one task for a simulated organism, with Box2D doing the physics and a GTK3 window showing the world and the learning graphs as it goes.

## Projects

Every folder in `projects/` is a task that gets compiled into its own executable, named after the folder. To add one of your own, see [projects/README.md](projects/README.md).

### `chase_target`

An organism in a box chases a target that moves around a circle. It pushes itself with four forces (up, down, left, right), and it can sense its distance to each wall, its offset from the target, and its own velocity. It learns as it goes with a linear actor-critic, and gets punished for hitting the walls and for using too much force. This is the Box2D port of `tests/test_3`.

### `find_food`

An agent in a box looks for food, but can only see so far. Its eight vision rays get folded into one vector pointing at the food. It gets rewarded for eating, and on every step for how close the food looks. The amount of food, the vision range, and the reward weights can all be changed from the GUI while it runs.

### `thruster_chase`

The same goal as `chase_target`, but the organism has to steer. All it has is a thruster on each leg that pushes along the way it is facing, so pushing harder on one side is the only way to turn.

### `island_food`

The thruster body from `thruster_chase` goes after food across a much bigger world with a group of islands. Food only turns up on land, and the target is always the closest piece. The organism can go in the water, but it is slow in there, and it costs score for as long as it stays.

### `memory_chase`

`chase_target`, except that the dot sometimes gets angry, and then being close to it is what gets punished. The organism can't see the mood of the dot. All it gets is a warning for the second before each change, so it has to remember the last warning for as long as the mood lasts. An LSTM actor-critic trains across 16 copies of the world at once (the first one is what gets drawn), and the angry spells slowly get longer to find out how long it can keep that up.

### `island_evolution`

`island_food` with 40 islands instead of 5, and a whole population instead of one organism. Each agent has an energy bar, which food fills up, and which living and thrusting use up. Once it runs out, the agent dies. Food grows back at a set rate, so the population settles at however many the islands can feed.

An agent with enough energy can reproduce, but it has to commit to it. Its network has a third output for how hard it is trying, which builds up progress towards a child at `dt * effort`, while slowing it down by up to 95%. Letting up loses the progress again, slowly. A child costs a set amount of energy, and gets 80% of it. It also gets its parent's network, put through the same mutations as CoDeepNEAT (new layers, new connections, new module species, layer sizes, activations, and the learning rate), along with whatever weights still fit. Every agent learns for itself as it goes, with its own actor-critic, and there is no selection apart from who lives long enough to have children. If everybody dies, the agent that ate the most over its life gets a child of its own to start over.

The founders are all the same: one small layer, with a skip around it. The panel on the World tab draws the network of the agent with the most children.

The agents are split between a fixed set of pools, one per core, each with a thread of its own, and a new agent joins whichever pool has the fewest. Sensing, running and training the networks, and scoring all happen in the pools at once, and a thread that finishes its own pool early helps with what is left of the others. The physics, eating, births, and deaths stay on the main thread, in order of id, so a run comes out exactly the same however many threads it has. On a 12 core Ryzen, that runs 2.7 to 4 times as fast as one thread.

### `continents`

`island_evolution`, grown into a whole world, so that agents can work with each other, or against each other. The map is 1000 by 700 meters, with six continents kept apart by at least 50 meters of sea, and a strip of land at each pole. It gets colder going north or south, and staying alive in the cold costs extra energy, less of it for bigger agents. The map is the same on every run.

The sea starts out calm, with a water friction of 2, so that the founders can spread out. Once every landmass has at least 5 agents on it at the same time, not counting founders, it gets rough, with a friction of 8, which mostly cuts the landmasses off from each other so that each population can go its own way. If everybody dies, it goes calm again. There is a slider for each.

The agents bump into each other now, and on top of thrusting and reproducing, each one has outputs to:

- **Bite** whatever is right in front of it, harder the bigger it is. Each bite takes energy out of the one bitten, and the biter gets 70% of it, which counts the same as eating, so that hunting pays off while it happens rather than only once the prey is dead and eaten. Health heals slowly. An agent that runs out of health dies the same as one that runs out of energy, and either way, what was left of its energy and its body gets scattered around as meat, which anybody can eat, and which rots after a while.
- **Hold on** to another agent in front of it, which joins them with a short rigid bar, so that it can drag it or push it around until it lets go, or gets pulled on harder than it is strong enough for.
- **Call out** on any of five channels. Everybody within 80 meters hears each channel as the total of everybody calling on it, getting quieter with the square of distance, along with which way it is coming from, on average.

Agents grow while their energy bar is over half full, and shrink while it is under. Growing costs energy, and shrinking gets most of it back. Bigger agents bite harder, take more to kill, hold more energy, and leave more meat behind, but need more food and are slower to turn.

Each agent sees through five eyes, which split up the view in front of it between them. Each eye sees how close the nearest food and the nearest agent are, how close the coast is, and for that agent, how big it is next to itself, and how much it looks like family. How wide the eyes look evolves. Every agent has markings, which are three numbers drawn as its colour, and a child gets its parent's markings with a little random drift. So close family look alike, and lines that have been apart for long, such as the ones on different continents, drift apart. Looking alike is the only way an agent has of telling friend from foe.

The networks evolve the same way as `island_evolution`, except that layers can be LSTMs as well as dense ones. Founders start with an LSTM, and a dense layer that sees it and the inputs, like `memory_chase`. To learn, an agent replays each 16 step stretch through its network from the memory it started that stretch with, so it doesn't have to keep every step's gradients around while it acts. Each action is judged by everything that came after it to the end of the stretch. A lambda slider can make that count less, and lean more on the critic's guess (generalised advantage estimation), but nothing tried below 1 learned any better. Another slider caps how random the bite can be. Left alone, its noise climbs to the most any output is allowed, so a bite is a coin flip, but capping it lower still didn't get agents to bite only what they touch. A third slider judges the bite on its own, by just what it got on the steps it could have bitten something on: what it took, what biting cost, and what it did to family. Turned all the way up, with the family harm slider turned up once the founders' families have drifted apart, agents in contact bite strangers more than close family in every seed tried, though only by a few percent. On its own, without the family harm, it makes agents bite nearly all the time. There are sliders for what everything costs and how everything is rewarded, including a share of nearby family's food reward, and a share of the harm an agent does to family by biting it (what the bitten one lost, times how related they are), which are both off to start with.

### `hail_mary`

A second go at what `continents` was after, rebuilt around what stopped it. Wolves live in packs that share one genome and one food store, and every weight in their brains is a gene, so evolution shapes what they do directly, and judges a whole life at once. Their prey are bison and hares, which run on fixed rules. A healthy bison runs as fast as a wolf and heals faster than one wolf can bite it, so it takes two or more, and grown bison gore wolves in front of them and charge wolves biting the herd. Wolves see 30 m, but hear calls on three channels from across the continent, packmates apart from strangers, so a wolf that finds a herd has something worth calling about. There are eight continents that wrap round at the edges, and only meet when a pack migrates, so half of them can have calls muted as a control. Hunting healthy bison together evolves on nearly every continent. Over 4 seeds, the continents that could hear calls were the ones that kept it up, killing about 1.7 times as many bison, while most of the muted ones fell back on hares, though one seed went the other way. See [projects/hail_mary/README.md](projects/hail_mary/README.md) for why each part is the way it is, the research behind it, and the results in full. `hail_mary_headless` runs it with no window, for experiments.

### `food_web`

The lessons of `hail_mary`, taken back to one walled map with four large continents, which anyone can swim between at a cost. Grass grows on the land, bison and hares eat it, and wolves eat them, and every one of the three evolves. Prey numbers come only from who is born and who dies: a continent only gets a wandering animal when it has almost none of a kind left. Bison, wolves, and hares rise and fall against each other every few hours, with wolves following bison. See [projects/food_web/README.md](projects/food_web/README.md).

## Installing the tools (MSYS2)

The build uses GCC, CMake, Ninja, and GTK3 from the MSYS2 **UCRT64** environment.

1. Install MSYS2 from [msys2.org](https://www.msys2.org/), in the default location of `C:\msys64`. `CMakePresets.json` has that path written in it, so installing it somewhere else means changing the paths in that file.

2. Open **MSYS2 UCRT64** from the Start menu. MSYS2 has several separate environments, each with its own packages, and the build only looks in `C:\msys64\ucrt64`. The shell prompt should say `UCRT64`.

3. Update MSYS2:

   ```bash
   pacman -Syu
   ```

   If that closes the terminal, open UCRT64 again and run `pacman -Su`.

4. Install the compiler, the build tools, and GTK3:

   ```bash
   pacman -S --needed git mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja mingw-w64-ucrt-x86_64-pkgconf mingw-w64-ucrt-x86_64-gtk3
   ```

   | Package | What it is for |
   | --- | --- |
   | `git` | Getting the code. Git for Windows works just as well. |
   | `mingw-w64-ucrt-x86_64-gcc` | `gcc` and `g++`. Also brings in binutils, whose `objcopy` is used to embed the Plotato icon. |
   | `mingw-w64-ucrt-x86_64-cmake` | CMake. A normal Windows install also works, as long as it is 3.24 or newer. |
   | `mingw-w64-ucrt-x86_64-ninja` | The generator that the CMake preset uses. |
   | `mingw-w64-ucrt-x86_64-pkgconf` | `pkg-config`, which is how CMake finds GTK. |
   | `mingw-w64-ucrt-x86_64-gtk3` | GTK3, along with Cairo, GLib, and the Adwaita icon theme. |

   These two are optional: GDB for debugging, and Glade for editing `static/gui.glade`.

   ```bash
   pacman -S --needed mingw-w64-ucrt-x86_64-gdb mingw-w64-ucrt-x86_64-glade
   ```

5. Check that everything is there. The last one should print a `3.24.x` version.

   ```bash
   g++ --version
   cmake --version
   ninja --version
   pkg-config --modversion gtk+-3.0
   ```

## Getting the code

Plotato and Box2D are submodules, so clone with `--recurse-submodules`:

```bash
git clone --recurse-submodules https://github.com/lganic/ADRIANNA.git
cd ADRIANNA
```

If you already cloned it without that, this fetches them:

```bash
git submodule update --init --recursive
```

To update an existing copy, along with its submodules:

```bash
git pull --recurse-submodules
```

## Compiling and running

Run all of these from the root of the repo, in the UCRT64 shell.

Configure first. This only needs doing once, and it finds new project folders by itself on the next build.

```bash
cmake --preset msys2-ucrt64
```

Then build a project, using the name of its folder, and run it:

```bash
cmake --build --preset msys2-ucrt64 --target chase_target
./build/chase_target.exe
```

The targets are `chase_target`, `find_food`, `thruster_chase`, `island_food`, `memory_chase`, `island_evolution`, `continents`, `hail_mary`, and `food_web`. To build all of them at once, leave off the target:

```bash
cmake --build --preset msys2-ucrt64
```

Executables have to run from the root of the repo, since they load `static/` from there.

The preset makes a Debug build, but the simulation code and its libraries always get compiled with `-O2`, so fast forward stays fast. To only configure some of the projects, add `"-DEVODEV_PROJECTS=find_food;chase_target"` to the configure command.

### From PowerShell or cmd

The executables need the GTK DLLs. The UCRT64 shell already has those on its path, but anywhere else needs `C:\msys64\ucrt64\bin` added first. In PowerShell:

```powershell
$env:Path = "C:\msys64\ucrt64\bin;$env:Path"
.\build\chase_target.exe
```

That only lasts for the one terminal. To keep it, add `C:\msys64\ucrt64\bin` to your user `Path` in the Windows environment variables.

`compile.bat` configures, builds, and runs a single project in one go. With no argument it runs `chase_target`. It reconfigures from scratch every time, so the commands above are quicker for repeat builds.

```powershell
.\compile.bat find_food
```

## Controls

These work in the window of every project:

| Key | What it does |
| --- | --- |
| `F` | Fast forward |
| `S` | Back to real time |
| Hold `V` | Watch in real time while fast forwarding |
| `Space` | Pause, and carry on again |
| `N` | Run one step, pausing first if it is running. Hold it to keep stepping |
| `[` / `]` | Slow motion: slower and faster, down to 1/32 of real time |
| Arrow keys | Move the view |
| `+` / `-` | Zoom in and out |

Pausing holds everything, fast forward included, and the view can still be moved and zoomed while it is paused. Pausing out of fast forward shows the exact step it stopped on, instead of the last frame that fast forward happened to show. Slow motion only spaces out the steps, which stay the same size, so it doesn't change what the simulation does. It applies whenever it is not fast forwarding, including while holding `V`.

## Python prototypes

`tests/` holds the early Python versions of the method. Each one is an organism in a box chasing a target, and ideas that work there get moved into the C++ code.

| Test | What it tries |
| --- | --- |
| `test_1` | A single linear matrix from inputs to forces, with its values tuned live. |
| `test_2` | A linear actor-critic that learns from TD error. |
| `test_3` | `test_2` with a natural gradient for the actor, scaled inputs, and pairwise features for the critic. `chase_target` is the port of this one. |

These use a normal Windows Python rather than the MSYS2 environment. Install the dependencies:

```bash
pip install pygame-ce numpy keyboard
```

Then run one from the root of the repo. They use relative imports, so they have to run as modules. `F`, `S`, and `V` work the same as they do in the GUI.

```bash
python -m tests.test_3.test
```

## Repo layout

| Path | What is in there |
| --- | --- |
| `projects/` | The tasks. |
| `src/sim/` | The framework that projects hook in to, and the actor-critic `sim::Organism`. |
| `src/gui/` | The GTK3 application that gets compiled into each project. |
| `src/cdn/` | CoDeepNEAT in C++, with its own autograd tensors. Documented in [src/cdn/docs/](src/cdn/docs/README.md). |
| `src/Plotato/` | Submodule. [Plotato](https://github.com/lganic/Plotato), the GTK3 plotting library. |
| `src/box2d/` | Submodule. [Box2D](https://github.com/erincatto/box2d), the physics engine. |
| `static/` | The Glade layout and CSS, loaded when a project starts. |
| `cmake/` | `evodev_add_project`, the function each project calls. |
| `tests/` | The Python prototypes. |

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Failed to load UI file`, or a theme error about `static\style.css` | The executable was started from the wrong folder. Run it from the root of the repo. |
| Windows says `libgtk-3-0.dll` was not found | `C:\msys64\ucrt64\bin` isn't on the path. See [From PowerShell or cmd](#from-powershell-or-cmd). |
| CMake says `src/Plotato` or `src/box2d` has no `CMakeLists.txt` | The submodules are missing. Run `git submodule update --init --recursive`. |
| CMake can't find `gtk+-3.0` or the compiler | The packages went in to a different MSYS2 environment. They need the `mingw-w64-ucrt-x86_64-` prefix. |

## License

MIT. See [LICENSE](LICENSE).
