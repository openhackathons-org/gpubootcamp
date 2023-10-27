
/* Copyright (c) 2012, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */



#include <pybind11/stl.h>
#include <thread>
#include <vector>
#include <cmath>

using namespace std;
namespace py = pybind11;
typedef tuple<float, float, float> centroid;

float compute_dist(centroid& p1, centroid& p2)
{
    float x1, y1, h1, x2, y2, h2;
    std::tie(x1, y1, h1) = p1;
    std::tie(x2, y2, h2) = p2;
    float dx = x2 - x1;
    float dy = y2 - y1;

    float lx = dx * 170 * (1/h1 + 1/h2) / 2;
    float ly = dy * 170 * (1/h1 + 1/h2) / 2;

    float l = sqrt(lx*lx + ly*ly);
    return l;
}

float compute_min_dist(int p, centroid& point, vector<centroid>& points) 
{
    vector<float> distances;
    for (auto & p2 : points) {
        distances.push_back(compute_dist(point, p2));
    }
    distances[p] = 1000000.0;
    float min_dist = *std::min_element(distances.begin(), distances.end());
    return min_dist;
}

vector<float> get_min_distances(vector<centroid>& points)
{
    vector<float> out;
    for (int p = 0; p < points.size(); p++) {
        float min_dist = compute_min_dist(p, points[p], points);
        out.push_back(min_dist);
    }
    return out;
}

PYBIND11_MODULE(distancing, m) {
    m.def("get_min_distances", &get_min_distances, "Get min distances");
}
