evodev_add_project(continents
    TITLE "Species evolving across continents"
    SEED 125
    SOURCES
        continents.cpp
        agent_pool.cpp
        pool_threads.cpp
        brain.cpp
        physics.cpp
        terrain.cpp
        organism_graphic.cpp
    LIBRARIES
        box2d
)
