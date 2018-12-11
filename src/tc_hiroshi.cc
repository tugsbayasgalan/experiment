// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <vector>
#include <fstream>
#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include "util.h"


/*
GAP Benchmark Suite
Kernel: Triangle Counting (TC)
Author: Scott Beamer

Will count the number of triangles (cliques of size 3)

Requires input graph:
  - to be undirected
  - no duplicate edges (or else will be counted as multiple triangles)
  - neighborhoods are sorted by vertex identifiers

Other than symmetrizing, the rest of the requirements are done by SquishCSR
during graph building.

This implementation reduces the search space by counting each triangle only
once. A naive implementation will count the same triangle six times because
each of the three vertices (u, v, w) will count it in both ways. To count
a triangle only once, this implementation only counts a triangle if u > v > w.
Once the remaining unexamined neighbors identifiers get too big, it can break
out of the loop, but this requires that the neighbors to be sorted.

Another optimization this implementation has is to relabel the vertices by
degree. This is beneficial if the average degree is high enough and if the
degree distribution is sufficiently non-uniform. To decide whether or not
to relabel the graph, we use the heuristic in WorthRelabelling.
*/


using namespace std;

size_t inline intersect_hiroshi(NodeID* A, NodeID* B, size_t totalA, size_t totalB, NodeID reference) {
    size_t begin_a = 0;
    size_t begin_b = 0;
    size_t count = 0;

    while (true) {
        NodeID Bdat0 = *(B + begin_b);
        NodeID Bdat1 = *(B + begin_b + 1);
        NodeID Bdat2 = *(B + begin_b + 2);
        //this ensures we are not double counting
        if (Bdat2 > reference) break;

        NodeID Adat0 = *(A + begin_a);
        NodeID Adat1 = *(A + begin_a + 1);
        NodeID Adat2 = *(A + begin_a + 2);

        if (Adat0 == Bdat2) {
            count++;
            goto advanceB; // no more pair
        }
        else if (Adat2 == Bdat0) {
            count++;
            goto advanceA; // no more pair
        }
        else if (Adat0 == Bdat0) {
            count++;
        }
        else if (Adat0 == Bdat1) {
            count++;
        }
        else if (Adat1 == Bdat0) {
            count++;
        }
        if (Adat1 == Bdat1) {
            count++;
        }
        else if (Adat1 == Bdat2) {
            count++;
            goto advanceB;
        }
        else if (Adat2 == Bdat1) {
            count++;
            goto advanceA;
        }
        if (Adat2 == Bdat2) {
            count++;
            goto advanceAB;
        }
        else if (Adat2 > Bdat2) goto advanceB;
        else goto advanceA;
        advanceA:
            begin_a += 3;
            if (begin_a >= totalA-2) { break; } else { continue; }
        advanceB:
            begin_b+=3;
            if (begin_b >= totalB-2) { break; } else { continue; }
        advanceAB:
            begin_a+=3; begin_b+=3;
            if (begin_a >= totalA-2 || begin_b >= totalB-2) { break; }
    }

    // intersect the tail using scalar intersection
  while (begin_a < totalA && begin_b < totalB) {

    //stops when w > v
    if(*(B + begin_b) > reference) break;

    if (*(A + begin_a) < *(B + begin_b)) {
      begin_a++;
    }
    else if (*(A + begin_a) > *(B + begin_b)) {
      begin_b++;
    }
    else {
      count++;
      begin_a++;
      begin_b++;
    }
  }
  return count;

}




size_t OrderedCountBinarySIMD(const Graph &g){
  size_t total = 0;
  #pragma omp parallel for reduction(+ : total) schedule(dynamic, 64)
  for(NodeID u = 0; u < g.num_nodes(); u++){
    size_t totalDegreeU = g.out_degree(u);
    for(NodeID v: g.out_neigh(u)){
      if (v > u) break;
      auto it_u = g.out_neigh(u).begin();
      auto ref = g.out_neigh(u).begin();
      auto end = g.out_neigh(u).end();
      auto it_v = g.out_neigh(v).begin();
      size_t totalDegreeV = g.out_degree(v);

      if (totalDegreeV > 100 && totalDegreeU > 100 && totalDegreeU > 0.05*totalDegreeV){
        total += intersect_hiroshi(it_u, it_v, totalDegreeU, totalDegreeV, v);
      }
      else {
        //Increments by 3
        for (NodeID w : g.out_neigh(v)) {
            if (w > v)
              break;

            while (*it_u < w){
              it_u += 3;
            }

            //if exceeds the boundary, set it at the boundary
            if (it_u >= end){
              it_u = end - 1;
            }

            if(*it_u == w){
              total++;
            }
            else {
              //rollback by 2 to make sure we are not skipping any intersections
              it_u -= 2;
              if(it_u >= ref){
                if(*it_u == w){
                  total++;
                }
              }
              it_u++;
              if(it_u >= ref){
                if(*it_u == w){
                  total++;
                }
              }
              it_u++;
            }
          }
      }

    }

  }
  return total;

}

size_t OrderedCount(const Graph &g) {
  size_t total = 0;
  #pragma omp parallel for reduction(+ : total) schedule(dynamic, 64)
  for (NodeID u=0; u < g.num_nodes(); u++) {
    for (NodeID v : g.out_neigh(u)) {
      if (v > u)
        break;
      auto it = g.out_neigh(u).begin();
      for (NodeID w : g.out_neigh(v)) {
        if (w > v)
          break;
        while (*it < w)
          it++;
        if (w == *it)
          total++;
      }
    }
  }
  return total;
}


// heuristic to see if sufficently dense power-law graph
bool WorthRelabelling(const Graph &g) {
  int64_t average_degree = g.num_edges() / g.num_nodes();
  if (average_degree < 10)
    return false;
  SourcePicker<Graph> sp(g);
  int64_t num_samples = min(int64_t(1000), g.num_nodes());
  int64_t sample_total = 0;
  pvector<int64_t> samples(num_samples);
  for (int64_t trial=0; trial < num_samples; trial++) {
    samples[trial] = g.out_degree(sp.PickNext());
    sample_total += samples[trial];
  }
  sort(samples.begin(), samples.end());
  double sample_average = static_cast<double>(sample_total) / num_samples;
  double sample_median = samples[num_samples/2];
  return sample_average / 1.3 > sample_median;
}


// uses heuristic to see if worth relabeling
size_t Hybrid(const Graph &g) {
  if (WorthRelabelling(g))
    return OrderedCountBinarySIMD(Builder::RelabelByDegree(g));
  else
    return OrderedCountBinarySIMD(g);
}


void PrintTriangleStats(const Graph &g, size_t total_triangles) {
  cout << total_triangles << " triangles" << endl;
}


// Compares with simple serial implementation that uses std::set_intersection
bool TCVerifier(const Graph &g, size_t test_total) {
  size_t total = 0;
  vector<NodeID> intersection;
  intersection.reserve(g.num_nodes());
  for (NodeID u : g.vertices()) {
    for (NodeID v : g.out_neigh(u)) {
      auto new_end = set_intersection(g.out_neigh(u).begin(),
                                      g.out_neigh(u).end(),
                                      g.out_neigh(v).begin(),
                                      g.out_neigh(v).end(),
                                      intersection.begin());
      intersection.resize(new_end - intersection.begin());
      total += intersection.size();
    }
  }
  total = total / 6;  // each triangle was counted 6 times
  if (total != test_total)
    cout << total << " != " << test_total << endl;
  return total == test_total;
}


int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "triangle count");
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Graph g = b.MakeGraph();
  if (g.directed()) {
    cout << "Input graph is directed but tc requires undirected" << endl;
    return -2;
  }
  BenchmarkKernel(cli, g, Hybrid, PrintTriangleStats, TCVerifier);
  return 0;


}
