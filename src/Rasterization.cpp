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

#include "Rasterization.h"
#include <queue>

// find height by scanning the nearest particles in the same row and column
double Rasterization::findHeightValByScanline(Particle *p, Cloth &cloth) {
  int xpos = p->pos_x;
  int ypos = p->pos_y;

  for (int i = xpos + 1; i < cloth.num_particles_width; i++) {
    double crresHeight = cloth.getParticle(i, ypos)->nearest_point_height;

    if (crresHeight > MIN_INF)
      return crresHeight;
  }

  for (int i = xpos - 1; i >= 0; i--) {
    double crresHeight = cloth.getParticle(i, ypos)->nearest_point_height;

    if (crresHeight > MIN_INF)
      return crresHeight;
  }

  for (int j = ypos - 1; j >= 0; j--) {
    double crresHeight = cloth.getParticle(xpos, j)->nearest_point_height;

    if (crresHeight > MIN_INF)
      return crresHeight;
  }

  for (int j = ypos + 1; j < cloth.num_particles_height; j++) {
    double crresHeight = cloth.getParticle(xpos, j)->nearest_point_height;

    if (crresHeight > MIN_INF)
      return crresHeight;
  }

  return findHeightValByNeighbor(p);
}

// find height by Region growing around the current particle
double Rasterization::findHeightValByNeighbor(Particle *p) {
  std::queue<Particle *> nqueue;
  std::vector<Particle *> pbacklist;
  int neighbor_size = p->neighborsList.size();

  // TODO RJ: this algorithm left the visited flag of some particles to "true"
  // it should be reseted to "false" after the algorithm is done because this
  // flag is reused in the cloth simultation. This is a bug with minor
  // consequences it seems to apply only to particles with id 0,1,
  // particle_number-1 and particle_number-2

  // initialize the queue with the neighbors of the current particle
  for (int i = 0; i < neighbor_size; i++) {
    p->is_visited = true;
    nqueue.push(p->neighborsList[i]);
  }

  // iterate over a queue of particle
  while (!nqueue.empty()) {
    Particle *pneighbor = nqueue.front();
    nqueue.pop();
    pbacklist.push_back(pneighbor);

    // if the current enqueued particle has a height defined, we return it
    if (pneighbor->nearest_point_height > MIN_INF) {

      // reset the visited flag for all the particles in the backlist
      for(auto p : pbacklist) {
        p->is_visited = false;
      };

      // reset the visited flag for all the particles in the queue
      while (!nqueue.empty()) {
        Particle *pp = nqueue.front();
        pp->is_visited = false;
        nqueue.pop();
      }

      // return the height value
      return pneighbor->nearest_point_height;

    } else { // else we schedule to visit the neighbors of the current neighbor
      for (auto ptmp : pneighbor->neighborsList) {
        if (!ptmp->is_visited) {
          ptmp->is_visited = true;
          nqueue.push(ptmp);
        }
      }
    }
  }

  return MIN_INF;
}

void Rasterization::Rasterize(Cloth &cloth, csf::PointCloud &pc,
                              std::vector<double> &heightVal) {

  for (std::size_t i = 0; i < pc.size(); i++) {
    const double pc_x = pc[i].x;
    const double pc_z = pc[i].z;

    const double deltaX = pc_x - cloth.origin_pos.f[0];
    const double deltaZ = pc_z - cloth.origin_pos.f[2];
    const int col = int(deltaX / cloth.step_x + 0.5);
    const int row = int(deltaZ / cloth.step_y + 0.5);

    if ((col >= 0) && (row >= 0)) {
      Particle *pt = cloth.getParticle(col, row);
      const double pc2particleDist =
          SQUARE_DIST(pc_x, pc_z, pt->getPos().f[0], pt->getPos().f[2]);

      if (pc2particleDist < pt->tmp_dist) {
        pt->tmp_dist = pc2particleDist;
        pt->nearest_point_height = pc[i].y;
      }
    }
  }

  heightVal.resize(cloth.getSize());
  for (int i = 0; i < cloth.getSize(); i++) {
    Particle *pcur = cloth.getParticle1d(i);
    const double nearestHeight = pcur->nearest_point_height;

    if (nearestHeight > MIN_INF) {
      heightVal[i] = nearestHeight;
    } else {
      heightVal[i] = findHeightValByScanline(pcur, cloth);
    }

  }
}
