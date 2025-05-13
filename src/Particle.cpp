// ======================================================================================
// Copyright 2017 State Key Laboratory of Remote Sensing Science,
// Institute of Remote Sensing Science and Engineering, Beijing Normal
// University

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ======================================================================================

#include "Particle.h"

/* This is one of the important methods, where the time is progressed
 *  a single step size (TIME_STEPSIZE) The method is called by
 *  Cloth.time_step() Given the equation "force = mass * acceleration"
 *  the next position is found through verlet integration*/
void Particle::timeStep()
{
    if (!movable) return;

    const double temp_height = height;
    height                   = height + (height - previous_height) * one_minus_damping + displacement;
    previous_height          = temp_height;
}

void Particle::satisfyConstraintSelf(uint32_t constraint_times)
{
    const double double_move_weight = (constraint_times > 14) ? 0.5 : doubleMove1[constraint_times];
    const double single_move_weight = (constraint_times > 14) ? 1.0 : singleMove1[constraint_times];

    for (Particle* neighbor_particle : neighborsList)
    {
        const double correction = neighbor_particle->height - height;

        if (isMovable() && neighbor_particle->isMovable())
        {
            const double delta = correction * double_move_weight;
            offsetPos(delta);
            neighbor_particle->offsetPos(-delta);
        }
        else if (isMovable())
        {
            offsetPos(correction * single_move_weight);
        }
        else if (neighbor_particle->isMovable())
        {
            neighbor_particle->offsetPos(-correction * single_move_weight);
        }
    }
}
