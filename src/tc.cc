// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <vector>
#include <fstream>
#include <emmintrin.h>
#include <x86intrin.h>
//#include <smmintrin.h>

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

long int inline BinarySearch(NodeID* it_begin, long int start, size_t total, NodeID target) {

  long int left = start == -1? 0 : start;
  long int right = total-1;
  while(left <= right) {

    long int medium = left + ((right - left) >> 1);
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

void print_m128(__m128i value, char c) {
    uint16_t *val = (uint16_t*) &value;
    for(int i = 0; i < 8; i++){
      cout << c << val[i];
      
    }
    cout << '\n';
    
}


uint16_t high16(uint32_t x) { return uint16_t(x >> 16); }


size_t intersect_32v2(NodeID* A, NodeID* B, size_t totalA, size_t totalB) {
  size_t count = 0;
  size_t begin_a = 0, begin_b = 0;
  // cout << "Begin intersect for " << totalA << " " << totalB << "\n";
  // for(int i = 0; i < totalA; i++){
  //   cout << "A" << *(A + i);
    
  // }
  // cout << "\n";
  // for(int i = 0; i < totalB; i++){
  //   cout << "B" << *(B + i);
    
  // }

  // cout << "End first" << "\n";

  size_t floorA = (totalA/4)*4;
  size_t floorB = (totalB/4)*4;

  while (begin_a < floorA && begin_b < floorB ) {
    __m128i v_a = _mm_loadu_si128((__m128i*)(A + begin_a));
    __m128i v_b = _mm_loadu_si128((__m128i*)(B + begin_b));

    __m128i res_v = _mm_cmpestrm(v_b, 4, v_a, 4,
    _SIDD_UWORD_OPS|_SIDD_CMP_EQUAL_ANY|_SIDD_BIT_MASK);

    int r = _mm_extract_epi32(res_v, 0);
    
    //cout << "FOund: " << _mm_popcnt_u32(r) << '\n';
    size_t a_last = _mm_extract_epi16(v_a, 3);
    size_t b_last = _mm_extract_epi16(v_b, 3);
    begin_a += ( a_last <= b_last ) * 4;
    begin_b += ( a_last > b_last ) * 4;
    
    count += _mm_popcnt_u32(r);

  }

    // intersect the tail using scalar intersection
  while (begin_a < totalA && begin_b < totalB) {

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

struct tempResult
{
     size_t count;
     size_t a_increment;
     size_t b_increment;
};

tempResult inline naive_comparison(NodeID* A, NodeID* B, size_t totalA, size_t totalB){
  size_t count = 0;
  size_t begin_a = 0;
  size_t begin_b = 0;
      // intersect the tail using scalar intersection
  while (begin_a < totalA && begin_b < totalB) {

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
  
  tempResult result = {count, begin_a, begin_b};
  return result;

}

size_t inline intersect_32(NodeID* A, NodeID* B, size_t totalA, size_t totalB) {
  size_t count = 0;
  size_t begin_a = 0, begin_b = 0;
  // cout << "Begin intersect for " << totalA << " " << totalB << "\n";
  // for(int i = 0; i < totalA; i++){
  //   cout << "A" << *(A + i);
    
  // }
  // cout << "\n";
  // for(int i = 0; i < totalB; i++){
  //   cout << "B" << *(B + i);
    
  // }

  // cout << "End first" << "\n";

  size_t floorA = (totalA/8)*8;
  size_t floorB = (totalB/8)*8;

  while (begin_a < floorA && begin_b < floorB ) {
    __m128i v_a_first= _mm_loadu_si128((__m128i*)(A + begin_a));
    __m128i v_b_first = _mm_loadu_si128((__m128i*)(B + begin_b));
    __m128i v_a_second= _mm_loadu_si128((__m128i*)(A + begin_a + 4));
    __m128i v_b_second = _mm_loadu_si128((__m128i*)(B + begin_b + 4));

    __m128i a_high = _mm_setr_epi16(high16(_mm_extract_epi32(v_a_first, 0)),
                                    high16(_mm_extract_epi32(v_a_first, 1)),
                                    high16(_mm_extract_epi32(v_a_first, 2)),
                                    high16(_mm_extract_epi32(v_a_first, 3)),
                                    high16(_mm_extract_epi32(v_a_second, 0)),
                                    high16(_mm_extract_epi32(v_a_second, 1)),
                                    high16(_mm_extract_epi32(v_a_second, 2)),
                                    high16(_mm_extract_epi32(v_a_second, 3)));
    __m128i b_high = _mm_setr_epi16(high16(_mm_extract_epi32(v_b_first, 0)),
                                    high16(_mm_extract_epi32(v_b_first, 1)),
                                    high16(_mm_extract_epi32(v_b_first, 2)),
                                    high16(_mm_extract_epi32(v_b_first, 3)),
                                    high16(_mm_extract_epi32(v_b_second, 0)),
                                    high16(_mm_extract_epi32(v_b_second, 1)),
                                    high16(_mm_extract_epi32(v_b_second, 2)),
                                    high16(_mm_extract_epi32(v_b_second, 3)));

    __m128i res_v_high = _mm_cmpestrm(b_high, 8, a_high, 8,
    _SIDD_UWORD_OPS|_SIDD_CMP_EQUAL_ANY|_SIDD_UNIT_MASK);

    if (!(bool)_mm_testz_si128(res_v_high,res_v_high)){

      tempResult val = naive_comparison(A+begin_a, B+begin_b, 8, 8);
      
      count += val.count;
      begin_a += val.a_increment;
      begin_b += val.b_increment;
    } else {
      size_t a_last = _mm_extract_epi16(v_a_second, 7);
      size_t b_last = _mm_extract_epi16(v_b_second, 7);
      begin_a += ( a_last <= b_last ) * 8;
      begin_b += ( a_last > b_last ) * 8;

    }
  }

    // intersect the tail using scalar intersection
  while (begin_a < totalA && begin_b < totalB) {

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

size_t intersect_16(NodeID* A, NodeID* B, size_t totalA, size_t totalB) {
  size_t count = 0;
  size_t begin_a = 0, begin_b = 0;
  // cout << "Begin intersect for " << totalA << " " << totalB << "\n";
  // for(int i = 0; i < totalA; i++){
  //   cout << "A" << *(A + i);
    
  // }
  // cout << "\n";
  // for(int i = 0; i < totalB; i++){
  //   cout << "B" << *(B + i);
    
  // }

  // cout << "End first" << "\n";

  size_t floorA = (totalA/8)*8;
  size_t floorB = (totalB/8)*8;

  while (begin_a < floorA && begin_b < floorB ) {
    __m128i v_a = _mm_loadu_si128((__m128i*)(A + begin_a));
    __m128i v_b = _mm_loadu_si128((__m128i*)(B + begin_b));
    print_m128(v_a, 'a');
    print_m128(v_b, 'b');

    __m128i res_v = _mm_cmpestrm(v_b, 8, v_a, 8,
    _SIDD_UWORD_OPS|_SIDD_CMP_EQUAL_ANY|_SIDD_BIT_MASK);
    print_m128(res_v, 'r');
    int r = _mm_extract_epi32(res_v, 0);
    
    //cout << "FOund: " << _mm_popcnt_u32(r) << '\n';
    size_t a_last = _mm_extract_epi16(v_a, 7);
    size_t b_last = _mm_extract_epi16(v_b, 7);
    begin_a += ( a_last <= b_last ) * 8;
    begin_b += ( a_last > b_last ) * 8;
    
    count += _mm_popcnt_u32(r);

  }

    // intersect the tail using scalar intersection
  while (begin_a < totalA && begin_b < totalB) {

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
      auto it_v = g.out_neigh(v).begin();
      size_t totalDegreeV = g.out_degree(v);
      total += intersect_32(it_u, it_v, totalDegreeU, totalDegreeV);

    }

   

    
  }
  return total/3;
  
}

// size_t OrderedCountBinary(const Graph &g){
//   size_t total = 0;
//   #pragma omp parallel for reduction(+ : total) schedule(dynamic, 64)
//   for (NodeID u=0; u < g.num_nodes(); u++) {
//     size_t totalDegreeU = g.out_degree(u);
//     for (NodeID v : g.out_neigh(u)) {
//       if (v > u)
//         break;
//       auto it = g.out_neigh(u).begin();
//       auto ref = g.out_neigh(u).begin();
//       auto end = g.out_neigh(u).end();
//       //This is multi increment
//       for (NodeID w : g.out_neigh(v)) {
//         if (w > v)
//           break;
//         // we increment by 2 to make sure we skip unneccesary checks 
//         while (*it < w){
//           it += 2;

//         }
//         // avoid out of bound problem
//         if (it >= end){
//           it = end - 1;
//         }


//         if(*it == w){
//           total++;
//         } 
//         else {
//           //roll back by 1
//           it--;
//           if(it >= ref){
//             if(*it == w){
//               total++;
//             }
//           }
//           it++;

//         }

//       }
//       }



    
//   }
 
//   return total;
  
// }

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
