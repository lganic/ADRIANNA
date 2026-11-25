# food_web

The lessons of `hail_mary`, taken back to one big map with continents on it. Grass grows, bison and hares eat it, and wolves eat them. How many of each there are comes from who is born and who dies, so they rise and fall against each other. Wolves, bison, and hares all evolve.

## What it keeps from hail_mary, and what it changes

| | hail_mary | food_web |
| --- | --- | --- |
| Map | 8 separate 400 m continents that wrap round | One walled 1800 by 1400 m sea, with four continents of .13 to .19 square km |
| Getting between continents | A pack migrating every 30 minutes | Swimming: anyone can, at 35% of top speed, for extra energy |
| Prey numbers | Topped up to 30 bison and 30 hares | Born when well fed, dead when killed, starved, or old |
| Prey food | None | Grass, which grows back on its own, on land only |
| Prey brains | Fixed rules | Evolved, like the wolves, each with its own mutated genome |
| Prey calls | None | One call each, heard by their own kind, on the same channel as the noise of one of them being bitten |
| Wander-ins | The main supply of prey | Only a rescue: a continent with fewer than 4 bison, 6 hares, or 1 pack gets one more |

Everything else is the same as `hail_mary`, which explains it: packs share one genome and one food store, every weight is a gene, founders start as plain animals rather than random brains, wolves see 30 m and hear packmates and strangers apart on 3 channels, a healthy bison is as fast as a wolf but heals faster than one can bite, and the noise of biting and feeding is on wolf channel 0. There is no muted control here.

## The world

**Grass** grows on every 4 m patch of land, back towards its most (.05 energy a square meter) logistically, so fastest when half eaten, and slowly back from bare ground too.

**Bison** live 1 to 2 hours, are grown at 15 minutes, and have a calf every 30 minutes while they are 85% full, which costs 35 of the 80 energy they can hold. Calves are small, and slower and weaker than grown bison: a newborn has half the strength of a bison in its prime (health, healing, and food in it), growing in to it by 15 minutes, and old bison fade back down to half over the last fifth of their lives. A hunted bison in its prime feeds 360.

**Hares** live 20 to 40 minutes, are grown at 5, and have a leveret every 8 minutes while 85% full. A hunted hare feeds 25.

**Wolves** live 1 to 2 hours. Packs of up to 8 have pups while they have food to spare, and full packs send their two youngest off as a daughter pack with a mutated genome.

**Water** slows everyone to 35% of top speed, and costs wolves and bison an extra .3 energy a second, and hares .05. Wolves and prey feel ahead for water, and founders turn away from it.

### What each kind senses

- **Wolves:** the same as `hail_mary`, plus whether they are in the water, whether there is water 10 m ahead, 45 degrees either side and straight on, and a random number of their own.
- **Bison and hares:** their own speed, turning, health, how full they are, pain, whether they are wet or young, and a random number. How much grass there is underfoot and 6 m off in three directions, and water 8 m off the same way. The two nearest wolves they can see, and the nearest of their own kind, how close and which way, and how many of each. And what they hear: their own kind calling, or being bitten, and wolves calling, biting, or feeding.

Founding bison and hares wander towards more grass, slow down to graze, turn away from water, turn away from the nearest wolf and run, and wander a little at random. Bison drift towards the nearest bison, which is enough to keep herds together. Nothing about facing up to wolves, or warning others, is there to start with.

## What it took

- **Hares bred themselves out of food, and starved by the thousand.** With grown up at 1 minute and young every minute, hares went from 60 to 465 a continent in 15 minutes, ate the grass to a quarter, and nearly everything starved. Life histories now go at a sensible pace for each other: grass in minutes, hares in tens of minutes, bison and wolves in hours.
- **Two grazers on one food can't both last.** Whichever can live on the least grass takes it all, and hares, small and quick to breed, could live on less than bison could, so bison starved to almost nothing. A bison now crops every patch under its muzzle and body, up to four at once, where a hare crops the one it stands on, the way big grazers can live on shorter grass than small ones. Bison now come out ahead, and hares do best when wolves have knocked the bison back.
- **A world wide rescue starved wolves of evolution.** With only 2 packs kept alive in the whole world, too few packs were alive at once for selection to work with. The rescue is per continent, like the prey's.
- **Wolves have to be able to grow on what they catch.** At 240 food a bison and 20 a hare, a wolf took in about as much as it spent, so packs never had pups, and the wolves alive were mostly ones the rescue had just put down. At 360 and 25, packs have pups and daughters, and rescues fall from about 20 an hour to under 10.

## What came of it

### With the first defaults

4 seeds, 72 simulated hours each, with calves every 15 minutes, at .4 of a grown bison's strength. That is 16 continents.

**Predator and prey cycle, though noisily.** Every continent goes up and down, over and over: bison between about 4 and 160, wolves between 2 and 47, and hares between 1 and 270, with 15 to 22 big swings of bison (a peak at least half as high again as the trough before it) in 66 hours, about every 3 hours. Wolves follow bison: on 12 of the 16 continents, wolf numbers line up best with bison numbers from a median of 1 hour before, and on all 16 the two go against each other at the same moment (median correlation -.28), which is the classic shape of predators chasing their prey's numbers. The correlations are weak, about .2, since four continents trade swimmers, and hares take up the slack whenever bison crash: hares boom when bison are low, since there is grass to spare, and wolves fall back on them.

**Wolves now live off what they catch.** About 1000 food a wolf an hour, 120 to 150 pups an hour across the world, and 20 to 26 daughter packs an hour. Rescues still add about a third of new wolves, since a continent whose wolves crash to nothing gets a pack again, about 4 an hour on each continent.

**Hunting together didn't evolve here, at least not in 72 hours.** Wolves biting each bison killed stayed at about 1.75 the whole time, and the share of kills that were weak bison went up a little, from .78 to .84. The likeliest reason is the prey: bison here are often short of grass (about 100 starve an hour), so weak ones are always easy to find, and there is nothing to gain by working together on a healthy one. Real wolves mostly take the weak too. In `hail_mary`, where every bison was healthy and grown when it arrived, hunting together paid, and evolved.

**Small changes in what evolved.** Wolves spent less of their time in the water as the runs went on, from 16% to 13%. None of the calls, wolf or prey, came to depend on anything in a consistent way.

### Getting wolves to hunt together

Splitting bison kills up showed that the easy food was calves: 50 to 75% of every bison killed, with old ones another 15 to 25%, and starving ones only 1 to 3%. And the longer a run went, the more wolves went for the weak: by 100 to 120 hours, 97% of kills were calves or old bison in the first seed.

Single 120 hour runs with less easy food:

| Calves | Every | What happened |
| --- | --- | --- |
| .4 strength (as before) | 15 minutes | Wolves turned to the weak more and more, and hunted in smaller groups |
| .6 strength | 15 or 30 minutes, with and without more grass | Too hard: wolves could hardly kill anything, and bison overran the grass |
| .4 strength | 30 minutes | Still mostly calves and old ones, 82 to 96% from 30 hours on |
| .5 strength | 15 minutes | A little more hunting together, but more weak kills too |
| .5 strength | 30 minutes | The only one where wolves hunted together more and more, and kept taking healthy grown bison. These are now the defaults |

Wolves hunt together more just because there are more of them about, so the fair comparison is at the same number of wolves on a continent. Fitting wolves biting each kill against the number of wolves, separately for the first and second half of each 120 hour run, and reading off both fits at 25 wolves on a continent, over 4 seeds each:

| Seed | Old defaults | New defaults |
| --- | --- | --- |
| 1 | 2.29 to 1.90 | 2.59 to 3.06 |
| 2 | 2.21 to 1.94 | 2.51 to 2.24 |
| 3 | 1.85 to 1.83 | 2.44 to 2.66 |
| 4 | 1.88 to 1.69 | 2.37 to 2.78 |

So with the old defaults, wolves came to hunt in smaller groups, in all 4 seeds, and with the new ones in bigger groups, in 3 of 4. The difference between them is .43 wolves a kill, but with 4 seeds each that could still be chance (permutation p .11). Over the whole world, wolves biting each kill went from 1.8 to 2.1 over the 120 hours with the new defaults, and from 1.8 to 1.7 with the old. Weak kills still went up, from .76 to .78 of the whole world's, though far less than with the old defaults (.77 to .85), and in the first seed 42% of kills were neither calves nor old by the end.

With wolves doing better, bison starved less (126 an hour to 52), and in two of the seeds, wolves held bison well below what the grass would feed, which left grass standing at .6 to .7 of its most. Predator and prey still cycled, with wolves following bison on 13 of the 16 continents.

In the first seed, bison also came to call louder with a wolf in sight than without (.21 louder by the end, from nothing), which is what an alarm call is. It didn't happen in the other three, so it may be chance.

What to try next: more seeds, to tell whether hunting together really evolves here, and longer runs; and no rescue for wolves at all, so that a continent's wolves can die out and come back only by swimming over.

## Running it

```bash
cmake --build --preset msys2-ucrt64 --target food_web
./build/food_web.exe
```

The map shows grass as green squares, darker the more there is, bison in brown, hares in grey, carcasses in dark red, and wolves coloured by pack. A blue ring is a wolf calling, an orange one is prey calling, and a red dot is a wolf biting. The World tab has a graph of bison, hares, and wolves over the last 24 hours on whichever continent the first slider picks, and a phase plot of wolves against bison, which goes round in loops when they cycle.

```bash
cmake --build --preset msys2-ucrt64 --target food_web_headless
./build/food_web_headless.exe --seed 1 --hours 72 --report 30 --threads 1
```

The harness prints a row for every continent, and one for the whole world, at each report: how many of each there were on average, grass, births, deaths of each kind, rescues, what wolves ate, hunting stats, calls, and how much of the time each kind spent in the water. Any slider can be set by part of its label with `--set "label=value"`, or part way through with `--at hours "label=value"`. `--draw N` draws everything every `N` steps without a window, and `--profile` prints where the time went.

A run comes out the same with any number of threads. Since physics and grass run on one thread anyway, several single threaded runs side by side are the most efficient way to run many: 4 at once ran at about 150 times real time each.
