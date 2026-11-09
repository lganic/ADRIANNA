evodev_add_project(thruster_chase
    TITLE "Chase the target with two thrusters"
    SEED 125
    SOURCES
        thruster_chase.cpp
        physics.cpp
        organism_graphic.cpp
    LIBRARIES
        box2d
)
