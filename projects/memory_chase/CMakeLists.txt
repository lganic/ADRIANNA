evodev_add_project(memory_chase
    TITLE "Chase the dot, unless it is angry"
    SEED 125
    SOURCES
        memory_chase.cpp
        memory_organism.cpp
        physics.cpp
    LIBRARIES
        box2d
)
