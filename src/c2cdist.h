#pragma once
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


#include "Cloth.h"
#include "PointCloud.h"

class c2cdist
{
   public:
    c2cdist(double threshold) : class_threshold(threshold) {}

    ~c2cdist() = default;

   public:
    void calCloud2CloudDist(
        const Cloth& cloth, const csf::PointCloud& pc, std::vector<int>& groundIndexes, std::vector<int>& offGroundIndexes);

   private:
    double class_threshold;
};
