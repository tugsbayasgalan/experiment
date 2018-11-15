// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <vector>
#include <fstream>
//#include <emmintrin.h>
//#include <x86intrin.h>
//#include <smmintrin.h>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include "inthash.h"
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
//function that computes the intersection of two sorted lists

long int inline BinarySearch(NodeID* it_begin, long int start, size_t total, NodeID target) {

  long int left = start == -1? 0 : start;
  long int right = total-1;
  while(left <= right) {

    long int medium = left + ((right - left) >> 2);
    NodeID current = *(it_begin + medium);

    if (current == target){
      return left;
    }

    if (current < target){
      left = medium + 1;
    } 

    else {
      right = medium - 1;
    }

  }

  return -1;


}



size_t OrderedCountBinary(const Graph &g){
  size_t total = 0;
  int skip_step = 2;
  #pragma omp parallel for reduction(+ : total) schedule(dynamic, 64)
  for (NodeID u=0; u < g.num_nodes(); u++) {
    size_t totalDegreeU = g.out_degree(u);
    for (NodeID v : g.out_neigh(u)) {
      if (v > u)
        break;
      auto it = g.out_neigh(u).begin();
      auto ref = g.out_neigh(u).begin();
      auto end = g.out_neigh(u).end();
      auto totalDegreeV = g.out_degree(v);
      
      if (totalDegreeU > 10000 && totalDegreeU < 0.01*totalDegreeV){
        int start = 0;
        int prevStart = 0;
        for (NodeID w : g.out_neigh(v)) {
          if (w > v)
            break;
          start = BinarySearch(it, start, totalDegreeU, w);
          
          if (start >= 0) {
            prevStart = start;
            total++;

          }
          else {
            start = prevStart;
          }

        }

      } 
      else {
        //This is multi increment
        for (NodeID w : g.out_neigh(v)) {
          if (w > v)
            break;

          while (*it < w){
            
            it += skip_step;

          }

          if (it >= end){
            it = end - 1;
          }


          if(*it == w){
            total++;
          } 
          else {
            it -= skip_step;
            int i = skip_step;
            while (i > 0){
              it--;
              i--;
              if(it >= ref){
                if(*it == w){
                  total++;
                  break;
                }
              }
            }

            it += skip_step-i;
  
          }

        }
    }
  }}
 
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
    return OrderedCountBinary(Builder::RelabelByDegree(g));
  else
    return OrderedCountBinary(g);
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
