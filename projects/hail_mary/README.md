# hail_mary

A second go at what `continents` was after: wolves that evolve to hunt together, and to call to each other, with a real-life counterpart for every part of the world they live in. This says what is different from `continents` and why, how to tell whether it worked, and what came of it.

## Why continents didn't get there

These come from the measurements in the continents runs, not guesses.

1. **Nothing could learn the rare things.** Weights only ever changed by each agent's own actor-critic, over a life of minutes. Contact with another agent was 1-2% of the time, and the reward for a bite was lost in the noise of everything else, so the bite never came to depend on what was in reach. Evolution couldn't fix it either, since mutation only changed structure, never weights.
2. **Nothing paid for working together.** Any one agent could bite anything, harder the bigger it was, so a second biter added nothing that the first couldn't. Food was everywhere, so nobody knew anything worth telling. And every agent was out for itself: self-interested agents don't use cheap talk ([Cao et al. 2018](https://arxiv.org/abs/1804.03980)), and robots only evolved honest signals when they were related, or selected as groups ([Floreano et al. 2007](https://infoscience.epfl.ch/record/99661)).
3. **Hunter and hunted were the same species,** so before any hunting could pay, agents had to learn not to eat their own family, which the learner never managed.
4. **Too much at once.** Cold, size, holding on, five channels, kin markings, and a 1000 by 700 m map all at the same time spread encounters thin, and made it hard to tell what did what.

## What this does instead

| Problem | Change | Real-life counterpart |
| --- | --- | --- |
| Rare events can't be learned in a life | Every weight is a gene, mutated in every daughter pack. Selection judges a whole life at once | Hunting instincts are mostly inherited. Gradient-free evolution of weights works at scale ([Such et al. 2017](https://arxiv.org/abs/1712.06567)) |
| Agents out for themselves | Wolves live in packs that share one genome and one food store | A wolf pack is a breeding pair and its young, sharing every kill. Robots evolve honest signals and altruism when related ([Floreano et al. 2007](https://infoscience.epfl.ch/record/99661), [Waibel et al. 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3086867)) |
| A second hunter adds nothing | A healthy bison heals faster (8/s) than one wolf bites (6/s), and runs as fast as a wolf, so it takes two or more | Wolves need more of them for bigger prey: elk success levels off at 2-6 wolves, bison at 9-13 ([MacNulty et al. 2012](https://digitalcommons.usu.edu/wild_facpub/1526/), 2014) |
| Nothing worth telling | Wolves see 30 m, but hear calls from across the continent | Wolves howl to reassemble, and ravens yell to bring others to a carcass ([Heinrich 1988](https://link.springer.com/doi/10.1007/BF00300349)). Communication evolves where one side knows what the other needs ([Werner and Dyer 1991](https://langev.com/author/gwerner)) |
| Signals have to start from nothing | Biting and feeding make noise on channel 0, the same channel a wolf can call on, whether it means to or not. Founders are just short of calling at all | Signals usually start out as cues that others already react to, and get exaggerated from there (ritualisation). Animals make all sorts of sounds before any of them mean anything |
| Same species as prey | Bison and hares run on fixed rules. Only wolves evolve | Wolves and elk, lions and buffalo, chimpanzees and colobus |
| A lone hunter can still take the weak | Calves, old, and hurt bison are slower and heal less. Grown bison charge anything biting the herd nearby, and gore anything in front of them | Wolves take mostly calves and old animals. Muskoxen and bison defend their calves |
| Walls do the cornering | Each continent wraps round at the edges | Open plain. On it, catching an animal as fast as you means cutting it off, or wearing it down |
| Too much at once | No size, cold, or holding on. Three channels. A 400 m continent | |

The `continents` idea of separate landmasses stays, as eight continents that only meet when a pack migrates from one to the next (every 30 minutes by default, and never in the experiments below). That is what lets half of them be run with calls muted, as the control for the other half.

### Things it took to get there

None of this worked first time, and how it failed says something about the problem:

- **Random brains never hunt.** In 12 hours, wolves with random weights never caught anything, so selection had nothing to go on. Founders now start as plain predators: they run at any hare or bison they see, bite what they catch, and eat carcasses. They take no notice of each other, of which way a bison is facing, or of sound, and never call. Hunting together starts out as a side effect of packmates all running at the same bison, which is where [Packer and Ruttan (1988)](https://scilit.com/publications/a46cf1f609a918d1e01ab27ccaf286e7) think it usually starts. Everything past that has to evolve.
- **Mutating a few weights a lot was far too slow.** Steering at something needs weights of about 2, and nudging 3% of weights by .1 got about one generation's worth of the way there in 25. Now every weight moves a little each time, by a size that evolves.
- **Coarse eyes couldn't aim.** Seeing which of five 54 degree slices something was in made wolves about a tenth as good as a fixed "run straight at it" rule. Each wolf now gets which way the nearest of each kind of thing is, exactly.
- **The first easy food is a trap.** With wolf carcasses edible, the first thing that evolved was packs killing and eating each other's wolves, which don't run away. Dead wolves now have nothing to eat.
- **Efficient predators eat their prey down to newborns.** With prey that breed on their own, wolves held bison at a handful of calves, which any wolf can catch alone, so there was nothing to cooperate on. Herds are now topped back up by grown bison wandering in, the way wolves don't decide how many caribou there are. Topping them up with calves instead just made the wolves farm calves.
- **Nobody knew anything the others didn't.** On a 240 m continent, a wolf could see prey 84% of the time, so there was nothing to call about. At 400 m across, with the same 30 bison in three herds, a wolf that finds a herd usually knows something its packmates don't.
- **Silent founders never start calling.** With founders' calls well below the point of making any sound, a call that meant something was hundreds of generations of drift away, and calling can't be selected for until somebody listens, which can't be selected for until somebody calls. Founders now sit just short of calling, and the noise of a kill is on channel 0, so going towards channel 0 can evolve first, because it means food, and calling on it can follow.
- **A full continent stopped evolving.** When there was only room for a few pups, whichever pack came first in the list got them, so the oldest packs had every one. The packs that have been eating best go first now, and full packs send their two youngest off to start a pack of their own, so packs keep turning over even when the continent is full.

## The world

Each continent is 400 m across and wraps round. It holds 30 bison and 30 hares, topped back up as they die.

- **Bison** graze in herds, and run from wolves they see within 30 m, or hear calling close by. A healthy grown one runs at 9 m/s, the same as a wolf, but is slower to turn and to get going. It has 80 health and heals 8 a second. Calves, old, and hurt ones run slower, heal slower, and have less to eat. Anything in front of its horns takes 8 damage a second (wolves have 30), a bitten bison sometimes turns on whoever bit it, and grown bison charge wolves biting the herd nearby. A hunted bison feeds 240.
- **Hares** run at 8 m/s, zigzag, and turn much faster than wolves. One wolf can catch one, and it feeds 12.
- **Carcasses** of bison that die of old age feed 60, and rot after two minutes.

## The wolves

A pack is a genome and a food store, with up to 6 wolves that are clones. Every wolf eats in to the store and lives off it. A pack with enough in store has a pup. A full pack with 400 in store sends its two youngest off as a new pack, with half of that, and the genome mutated. Wolves live 20 to 40 minutes, and starve or die of their wounds sooner. Up to 60 wolves fit on a continent.

Each wolf has a body (9 m/s, 270 degree view), and a brain of 24 leaky recurrent units, which can remember a few seconds. It doesn't learn during its life: everything it does is inherited.

It senses:

- Its own speed, turning, health, the pack's store, pain, whether a bison, hare or stranger is in reach of a bite, and whether it is eating.
- For the nearest hare, bison, carcass, two packmates, and wolf from another pack: how close, and which way. For the bison, also which way it is facing and how weak it is. And how many packmates it can see.
- For each of 3 channels, sound from packmates and from strangers apart: how loud in total, on a log scale, and which way it comes from. Channel 0 also carries the noise of biting and feeding, at .3 of a full volume call.

It decides how fast to go, which way to turn, whether to bite, and how loud to call on each channel. Calling costs energy, and prey within about 20 m hear it.

Muting a continent silences calls, for wolves and prey alike, but not the noise of biting and feeding, so a muted continent still has cues, just no signals.

## How to tell whether it worked

Watching it isn't enough, since agents can look like they communicate when they don't ([Lowe et al. 2019](https://arxiv.org/abs/1903.05168)). So there are three kinds of measure:

- **Hunting together:** bison kills an hour, how many of them were weak, how many wolves bit each one in its last 10 seconds, how evenly round it they were at the end (1 for all on one side, 0 for all the way round), and how often wolves bite from in front of the horns, or get gored.
- **Signalling:** whether calls depend on what is going on. For each channel, how much louder a wolf calls at or in sight of a bison than with nothing in sight, taking whichever channel differs the most.
- **Listening:** whether wolves go towards packmates' calls. Over every step a wolf hears a packmate's call loudly (as loud as a full call from about 70 m) with nothing else in sight, its speed towards the call, as a fraction of top speed. On muted continents the same thing is worked out from what would have been heard, so the two can be compared. The harness also gives this for each channel on its own, and for the noise of packmates biting and feeding, along with how far each wolf is from its nearest packmate.

The real test is causal. Half the continents have calls muted from the start: calls are made and paid for, but nobody hears them. If calling does anything, the continents that can hear should hunt better, and muting them at the end should make them hunt worse.

## What came of it

4 seeds of 8 continents each, for 100 simulated hours, with no migration. The odd continents were muted from the start, and at 90 hours every continent was muted. That is 16 continents that could hear against 16 that couldn't. The p values are from a two-sided permutation test on each continent's average over hours 50 to 90.

**Hunting together evolves, on nearly every continent.** Founders already bring down some bison by all running at the same one, but mostly calves and old ones. Across every continent, from the first 5 hours to hours 30 to 60, bison kills an hour went from 105 to 227, the share of food from bison from .46 to .66, the share of kills that were healthy grown bison (which take more than one wolf) from 31% to 52%, and the wolves biting each kill from 2.8 to 3.2. The last bites also came from further round the bison: the surround measure fell from .86 to .73. 15 of the 16 continents that could hear, and 12 of the 16 that couldn't, got to the point of filling up with wolves on the back of it. The fixed rule, which runs straight at the nearest prey with perfect aim, can't do that at all: it holds about 17 wolves on 620 food an hour each, killing 125 bison an hour, half of them weak, with 2.6 wolves to a kill. A thriving evolved continent holds 60 on about 1300 each.

**Each continent ends up in one of two states.** Either thriving, at or near the 60 wolf limit, killing 300 to 500 bison an hour, two thirds of them grown, or collapsed, at about 25 wolves living mostly on hares, calves, and old bison. Continents switch from one to the other and back, so any one continent's numbers jump around, and only the counts over many continents mean much.

**Continents that could hear calls kept hunting bison. The muted ones mostly lost it.** For the first 25 hours the two kinds did about the same, and muted ones were a little ahead, maybe because random calls warned the prey on the continents that could hear them. After that they came apart:

| Hours 50 to 90, per continent | Heard (16) | Muted (16) | p |
| --- | --- | --- | --- |
| Bison killed an hour | 239 | 141 | .03 |
| Wolves alive | 43 | 34 | .03 |
| Share of food from bison | .73 | .52 | .04 |
| Wolves biting each bison killed | 3.4 | 2.6 | .04 |
| Wolves gored an hour | 54 | 30 | .02 |
| Share of reports thriving (50+ wolves), hours 60 to 90 | .40 | .12 | |

That is a stag hunt: hunting big prey together pays the most, but going after hares alone is the safe fallback, and populations can slide from one to the other. Without anything to signal with, the muted continents slid to hares, and stayed there. In game theory, even signals that cost nothing change which of the two a population of stag hunters ends up in ([Skyrms 2004](https://www.cambridge.org/9780521533928), chapter 5), and this looks like the same thing happening in an evolved ecology rather than a game.

It isn't a sure thing, though. In 3 of the 4 seeds the heard continents were clearly ahead from about 45 hours on. In the fourth, the muted ones did better for most of the run, and the single best continent of the whole experiment (about 500 bison an hour, 99% of its food from bison) was a muted one. So hunting together doesn't strictly need calls, the same as [Muro et al. (2011)](https://czaw.org/?p=7596) found: being able to hear makes it more likely to last.

**What the calls are, is harder to say.** In 3 of 4 seeds, channel 0, the one that also carries the noise of biting and feeding, was louder at or in sight of a bison than with nothing in sight: by 2 to 3 times in two of them. That is what ritualisation should look like, a cue turning in to a signal. The other channels are used differently in every seed. Muted wolves call just as much, and their calls depend on what is going on just as much, since nothing stops them: with nobody hearing, calling costs only a little energy. So which situations a wolf calls in says nothing on its own, the same as [Lowe et al. (2019)](https://arxiv.org/abs/1903.05168) warned.

**The noise of a kill brings packmates in, everywhere.** With nothing in sight, wolves run towards the sound of a packmate biting or feeding, at .26 to .73 of top speed on all 8 continents of seed 1, muted or not. That cue is the recruitment that clearly evolved.

**Hearing calls changes how wolves move, but not in one direction.** Wolves that could hear went away from a packmate's call faster than muted wolves did from where the call would have been (-.18 against -.11 of top speed, p .04), but in seed 3 it was the other way round. Both are negative because of what the measure picks out: a wolf that can't see any of its packmates, but can hear one, is usually one that is drifting away from them. Founders, which take no notice of calls at all, already score -.36.

**Taking calls away for the last 10 hours did little on average.** The 5 heard continents that were thriving when calls were muted lost 1% to 36% of their bison kills over the next 10 hours, 17% on average, while over every heard continent there was no change beyond the noise. So calls mattered more over evolutionary time, in which continents kept hunting bison, than from moment to moment.

**What this doesn't show.** Nothing here shows wolves taking up different roles in a hunt, or calling to bring in packmates from out of sight, on any consistent basis. The control for relatedness (packs of 1, which turns kin selection off) hasn't been run, and its reproduction would need rebalancing first, since a lone wolf can't split off a pair. And four seeds is a small sample for an outcome that is this bimodal per continent.

## Running it

```bash
cmake --build --preset msys2-ucrt64 --target hail_mary
./build/hail_mary.exe
```

The first slider picks which continent is drawn. Wolves are coloured by pack, and daughter packs look like their mothers. A ring round a wolf is a call, in the colour of its loudest channel. Red jaws are a bite.

`hail_mary_headless` runs it with no window, and prints a row for every continent at each report, then one for the heard and one for the muted continents together:

```bash
cmake --build --preset msys2-ucrt64 --target hail_mary_headless
./build/hail_mary_headless.exe --seed 1 --hours 100 --report 300 --set "migrations=0" --set "Calls muted=1" --at 90 "Calls muted=2" --calls 70 90
```

That is exactly one seed of the experiment below. Any slider can be set by part of its label. The heard and muted rows always mean the continents that started out that way, even after `--at` changes the mute slider.

- `--calls H1 H2` prints how loud each channel was in each situation, between hours `H1` and `H2`.
- `--trace H S C file.json` records `S` seconds of continent `C` from hour `H`: every animal and carcass, five times a second, with who is biting and calling.
- `--draw N` draws every `N` steps on a graph that never goes on screen, to check the drawing.
- `--threads N` makes no difference to what happens: runs come out bit for bit the same with any number of threads.

A run takes about 15 minutes for 100 simulated hours on 12 cores. Don't run several at once: the threads spin while they wait for each other, so runs side by side slow each other down to a crawl. Run them one after another, and from a copy of the executable, since Windows won't let a running one be rebuilt.

There is also a slider that replaces every brain with the fixed "run at the nearest prey" rule, which is the yardstick that evolution has to beat.
