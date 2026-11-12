evodev_add_project(island_evolution
    TITLE "Thrusters evolving across an archipelago"
    SEED 125
    SOURCES
        island_evolution.cpp
        agent_pool.cpp
        pool_threads.cpp
        brain.cpp
        physics.cpp
        organism_graphic.cpp
    LIBRARIES
        box2d
)
