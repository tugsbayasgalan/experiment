// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <assert.h>

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
#define USEHASH 1
#define CHUNK_SIZE 32
#define PAR_THRESHOLD 500

// julian's hash table
typedef ETable<hashInt<uint>, uintT> intTable;


using namespace std;


// size_t OrderedCountHashJulian(const Graph &g){
//   int64_t num_nodes = g.num_nodes();
//   //compute hash table offsets
//   uintT* hoffsets = newA(uintT,num_nodes+1);
//   parallel_for(uintT i=0;i<num_nodes;i++) {
//     auto size = g.out_neigh(i).end() - g.out_neigh(i).begin();
//     hoffsets[i] = 1 << (utils::log2Up((uintT)(size + 1))); 
//   }
//   hoffsets[num_nodes] = 0;
//   //creating one big array for all hashes
//   uintT totalSize = sequence::plusScan(hoffsets,hoffsets,num_nodes+1);
//   uint* A = newA(uint,totalSize);

//   intTable* TA = newA(intTable,num_nodes);
//   #pragma omp parallel for schedule(static, CHUNK_SIZE)
//   for(long s=0;s<num_nodes;s++) {
//     uintT size = hoffsets[s+1]-hoffsets[s];
//     TA[s] = intTable(size, hashInt<uint>(), A+hoffsets[s]);
//   }
  

//   //everyone inserts neigbors
//   #pragma omp parallel for schedule(static, CHUNK_SIZE)
//   for (uintT s=0;s<num_nodes;s++) {
//     uintT d = hoffsets[s];
//     if(d > 10000) {
//       #pragma omp parallel for schedule(static, CHUNK_SIZE)
//       for(auto j = g.out_neigh(s).begin(); j < g.out_neigh(s).end(); j++){
//         TA[s].insert(*j);
//       }

//     }
//     else {
//       for(auto j = g.out_neigh(s).begin(); j < g.out_neigh(s).end(); j++){
//         TA[s].insert(*j);
//       }
//     }   
//   }
//   size_t total = 0;
//   #pragma omp parallel for reduction(+ : total) schedule(dynamic, CHUNK_SIZE)
//   for (NodeID u=0; u < g.num_nodes(); u++) {
//     //if degree > 10000 execute parallel
//     size_t currentCount;
//     if (g.out_degree(u) > 10000) {
//       //to keep track of count for each node (this will be reduced)
//       uint countArray[g.out_degree(u)];
//       #pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
//       for (NodeID v : g.out_neigh(u)) {
//         if (v > u) 
//           //figure out how break works
//           break;
        
//         for (NodeID w : g.out_neigh(v)) {
//           if (w > v)
//             break;
//           if (TA[u].find(w)){
//               countArray[v]++;
//           }
//         }
//       }
//       currentCount = sequence::plusScan(countArray,countArray,g.out_degree(u)+1);
//       free(countArray);
//     } 
//     else {
//       size_t localCount = 0;
//       for (NodeID v : g.out_neigh(u)) {
//         if (v > u) {
//           //figure out how break works
//           break;
//         }

//         for (NodeID w : g.out_neigh(v)) {
//           if (w > v)
//             break;

//           if (TA[u].find(w)){
//               localCount++;
//           }
//         }
//       }
//       currentCount = localCount;
//     }

//     total += currentCount;
 
//   }

//   free(TA);
//   free(A);

//   return total;

// }


size_t OrderedCountHashJulian(const Graph &g){
  int64_t num_nodes = g.num_nodes();
  //compute hash table offsets
  uintT* hoffsets = newA(uintT,num_nodes+1);
  parallel_for(uintT i=0;i<num_nodes;i++) {
    auto size = g.out_neigh(i).end() - g.out_neigh(i).begin();
    hoffsets[i] = 1 << (utils::log2Up((uintT)(size + 1))); 
  }
  hoffsets[num_nodes] = 0;
  //creating one big array for all hashes
  uintT totalSize = sequence::plusScan(hoffsets,hoffsets,num_nodes+1);
  uint* A = newA(uint,totalSize);

  intTable* TA = newA(intTable,num_nodes);
  #pragma omp parallel for schedule(dynamic, 64)
  for(long s=0;s<num_nodes;s++) {
    uintT size = hoffsets[s+1]-hoffsets[s];
    TA[s] = intTable(size, hashInt<uint>(), A+hoffsets[s]);
  }
  

  //everyone inserts neigbors
  #pragma omp parallel for schedule(dynamic, 64)
  for (uintT s=0;s<num_nodes;s++) {
    uintT d = hoffsets[s];
    if(d > 10000) {
      #pragma omp parallel for schedule(dynamic, 64)
      for(auto j = g.out_neigh(s).begin(); j < g.out_neigh(s).end(); j++){
        TA[s].insert(*j);
      }

    }
    else {
      for(auto j = g.out_neigh(s).begin(); j < g.out_neigh(s).end(); j++){
        TA[s].insert(*j);
      }
    }
    
      
  }
  free(hoffsets);
  size_t total = 0;
  #pragma omp parallel for reduction(+ : total) schedule(dynamic, 64)
  for (NodeID u=0; u < g.num_nodes(); u++) {
    if (g.out_degree(u) > PAR_THRESHOLD) {
      size_t localCount = 0;
      auto v = g.out_neigh(u).begin();
      #pragma omp parallel private(v) reduction(+ : localCount) 
      {
        for (v = g.out_neigh(u).begin(); v != g.out_neigh(u).end(); v++) {
        //for (NodeID v : g.out_neigh(u)) {

            //else {
                //if v is greater than u, we want to exit (used flag since openmp doesn't like break)
          if (*v <= u) {
            for (NodeID w : g.out_neigh(*v)) {
              if (w > *v)
                break;
              if (TA[u].find(w)){ 
                  localCount++;
              }
            }

          }
                  //#pragma omp critical
                  //flag = true;
                

            //}
          
          
        }
      }

      total += localCount;

    } 
    else {
      size_t localCount = 0;
      for (NodeID v : g.out_neigh(u)) {
        if (v > u)
          break;
        for (NodeID w : g.out_neigh(v)) {
          if (w > v)
            break;

          if (TA[u].find(w)){
              localCount++;
          }
        }
      }

      total += localCount;
    }

  }
  free(TA);
  free(A);
  return total;




 
}



size_t OrderedCountHash(const Graph &g) {

  unordered_map<NodeID, unordered_map<NodeID, int>> neighbor_tables = unordered_map<NodeID, unordered_map<NodeID, int>>();
  size_t total = 0;
  int64_t num_nodes = g.num_nodes();
  // #pragma omp parallel
  // {

    
  //   unordered_map<NodeID, unordered_map<NodeID, int>> local_tables;
  //   for(NodeID i=0; i < num_nodes; i++){

  //     //neighbor_tables[i] = unordered_map<NodeID, int>();
  //     unordered_map<NodeID, int> u_neighbors;
  //     #pragma omp parallel for schedule(dynamic, 1024)
  //     for(auto j = g.out_neigh(i).begin(); j < g.out_neigh(i).end(); j++){
  //       u_neighbors.insert(pair <NodeID, int> (*j, 1));
  //     }

    
  //     local_tables.insert(pair<NodeID, unordered_map<NodeID, int>>(i, u_neighbors));

  //   }

  //   #pragma omp critical
  //   neighbor_tables.insert(local_tables.begin(), local_tables.end());



  // }

  #pragma omp parallel for schedule(dynamic, 64)
  for(NodeID i=0; i < num_nodes; i++){

    //neighbor_tables[i] = unordered_map<NodeID, int>();
    unordered_map<NodeID, int> u_neighbors;
    //#pragma omp parallel for reduction(+ : total) schedule(dynamic, 1024)
    for(auto j = g.out_neigh(i).begin(); j < g.out_neigh(i).end(); j++){
      u_neighbors.insert(pair <NodeID, int> (*j, 1));
    }

    #pragma omp critical
    neighbor_tables.insert(pair<NodeID, unordered_map<NodeID, int>>(i, u_neighbors));
  
  }

  #pragma omp parallel for reduction(+ : total) schedule(dynamic, 64)
  for (NodeID u=0; u < g.num_nodes(); u++) {
  
    for (NodeID v : g.out_neigh(u)) {
      if (v > u)
        break;
      for (NodeID w : g.out_neigh(v)) {
        if (w > v)
          break;

        if (neighbor_tables[u].find(w) != neighbor_tables[u].end()){
            total++;
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

  if (WorthRelabelling(g)) {
      #if USEHASH==1
          return OrderedCountHashJulian(Builder::RelabelByDegree(g));
      #else
          return OrderedCount(Builder::RelabelByDegree(g));
      #endif

  } else {
      #if USEHASH==1
        return OrderedCountHashJulian(g);
      #else
        return OrderedCount(g);
      #endif

  }

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
