/* 
 * Copyright 2016 Emaad Ahmed Manzoor
 * License: Apache License, Version 2.0
 * http://www3.cs.stonybrook.edu/~emanzoor/streamspot/
 */

#include <algorithm>
#include <bitset>
#include <cassert>
#include <chrono>
#include <deque>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cluster.h"
#include "docopt.h"
#include "graph.h"
#include "hash.h"
#include "io.h"
#include "param.h"
#include "simhash.h"
#include "streamhash.h"

using namespace std;

static const char USAGE[] =
R"(StreamSpot.

    Usage:
      streamspot --edges=<edge file>
                 --chunk-length=<chunk length>

      streamspot (-h | --help)

    Options:
      -h, --help                              Show this screen.
      --edges=<edge file>                     Incoming stream of edges.
      --chunk-length=<chunk length>           Parameter C.
)";

void allocate_random_bits(vector<vector<uint64_t>>&, mt19937_64&, uint32_t);
void compute_similarities(const vector<shingle_vector>& shingle_vectors,
                          const vector<bitset<L>>& simhash_sketches,
                          const vector<bitset<L>>& streamhash_sketches);
void construct_random_vectors(vector<vector<int>>& random_vectors,
                              uint32_t rvsize,
                              bernoulli_distribution& bernoulli,
                              mt19937_64& prng);
void construct_simhash_sketches(const vector<shingle_vector>& shingle_vectors,
                                const vector<vector<int>>& random_vectors,
                                vector<bitset<L>>& simhash_sketches);
void perform_lsh_banding(const vector<uint32_t>& normal_gids,
                         const vector<bitset<L>>& simhash_sketches,
                         vector<unordered_map<bitset<R>,vector<uint32_t>>>&
                            hash_tables);
void print_lsh_clusters(const vector<uint32_t>& normal_gids,
                        const vector<bitset<L>>& simhash_sketches,
                        const vector<unordered_map<bitset<R>,vector<uint32_t>>>&
                            hash_tables);
void test_anomalies(uint32_t num_graphs,
                    const vector<bitset<L>>& simhash_sketches,
                    const vector<unordered_map<bitset<R>,vector<uint32_t>>>&
                      hash_tables);

int main(int argc, char *argv[]) {
  vector<vector<uint64_t>> H(L);                 // Universal family H, contains
                                                 // L hash functions, each
                                                 // represented by chunk_length+2
                                                 // 64-bit random integers

  mt19937_64 prng(SEED);                         // Mersenne Twister 64-bit PRNG
  bernoulli_distribution bernoulli(0.5);         // to generate random vectors
  vector<vector<int>> random_vectors(L);         // |S|-element random vectors
  unordered_map<string,uint32_t> shingle_id;
  unordered_set<string> unique_shingles;
  //vector<unordered_map<bitset<R>,vector<uint32_t>>> hash_tables(B);

  // for timing
  chrono::time_point<chrono::steady_clock> start;
  chrono::time_point<chrono::steady_clock> end;
  chrono::nanoseconds diff;

  // arguments
  map<string, docopt::value> args = docopt::docopt(USAGE, { argv + 1, argv + argc });

  string edge_file(args["--edges"].asString());
  uint32_t chunk_length = args["--chunk-length"].asLong();

  cerr << "StreamSpot Graphs-to-Shingle-Vectors (";
  cerr << "C=" << chunk_length << "";
  cerr << ")" << endl;

  // FIXME: Tailored for this configuration now
  assert(K == 1);

  uint32_t num_graphs;
  vector<edge> train_edges;
  cerr << "Reading training edges..." << endl;
  start = chrono::steady_clock::now();
  tie(num_graphs, train_edges) = read_edges(edge_file);
  end = chrono::steady_clock::now();
  diff = chrono::duration_cast<chrono::nanoseconds>(end - start);
  cerr << "\tReading edges took: ";
  cerr << static_cast<double>(diff.count()) << "us" << endl;

  // per-graph data structures
  vector<graph> graphs(num_graphs);
  vector<shingle_vector> shingle_vectors(num_graphs);

  // construct training graphs
  cerr << "Constructing " << num_graphs << " training graphs..." << endl;
  start = chrono::steady_clock::now();
  for (auto& e : train_edges) {
    update_graphs(e, graphs);
  }
  end = chrono::steady_clock::now();
  diff = chrono::duration_cast<chrono::nanoseconds>(end - start);
  cerr << "\tGraph construction took: ";
  cerr << static_cast<double>(diff.count()) << "us" << endl;

  // construct shingle vectors
  cerr << "Constructing shingle vectors:" << endl;
  start = chrono::steady_clock::now();
  construct_shingle_vectors(shingle_vectors, shingle_id, graphs, chunk_length);
  end = chrono::steady_clock::now();
  diff = chrono::duration_cast<chrono::nanoseconds>(end - start);
  cerr << "\tShingle vector construction took: ";
  cerr << static_cast<double>(diff.count()) << "us" << endl;

  // print shingles
  cout << shingle_id.size() << "\t" << chunk_length << endl;
  vector<string> shingles(shingle_id.size());
  for (auto& kv : shingle_id) {
    shingles[kv.second] = kv.first;
  }

  cout << "shingles" << "\t";
  for (uint32_t i = 0; i < shingles.size() - 1; i++) {
    cout << shingles[i] << "\t";
  }
  cout << shingles[shingles.size() - 1] << endl;

  // print shingle vectors
  for (uint32_t i = 0; i < num_graphs; i++) {
    cout << i << "\t"; // graph id
    for (uint32_t j = 0; j < shingle_id.size() - 1; j++) {
      cout << shingle_vectors[i][j] << "\t"; // shingle frequency
    }
    cout << shingle_vectors[i][shingle_id.size() - 1] << endl;
  }

  return 0;
}

void allocate_random_bits(vector<vector<uint64_t>>& H, mt19937_64& prng,
                          uint32_t chunk_length) {
  // allocate random bits for hashing
  for (uint32_t i = 0; i < L; i++) {
    // hash function h_i \in H
    H[i] = vector<uint64_t>(chunk_length + 2);
    for (uint32_t j = 0; j < chunk_length + 2; j++) {
      // random number m_j of h_i
      H[i][j] = prng();
    }
  }
#ifdef DEBUG
    cout << "64-bit random numbers:\n";
    for (int i = 0; i < L; i++) {
      for (int j = 0; j < chunk_length + 2; j++) {
        cout << H[i][j] << " ";
      }
      cout << endl;
    }
#endif
}

void compute_similarities(const vector<shingle_vector>& shingle_vectors,
                          const vector<bitset<L>>& simhash_sketches,
                          const vector<bitset<L>>& streamhash_sketches) {
  for (uint32_t i = 0; i < shingle_vectors.size(); i++) {
    for (uint32_t j = 0; j < shingle_vectors.size(); j++) {
      double cosine = cosine_similarity(shingle_vectors[i],
                                        shingle_vectors[j]);
      double angsim = 1 - acos(cosine)/PI;
      double simhash_sim = simhash_similarity(simhash_sketches[i],
                                              simhash_sketches[j]);
      double streamhash_sim = streamhash_similarity(streamhash_sketches[i],
                                                    streamhash_sketches[j]);
      cout << i << "\t" << j << "\t";
      cout << cosine;
      cout << "\t" << angsim;
      cout << "\t" << simhash_sim << "," << cos(PI*(1.0 - simhash_sim)) << " ";
      cout << "\t" << streamhash_sim << "," << cos(PI*(1.0 - streamhash_sim)) << " ";
      cout << "\t" << (streamhash_sim - angsim);
      cout << endl;
    }
  }
}

void construct_random_vectors(vector<vector<int>>& random_vectors,
                              uint32_t rvsize,
                              bernoulli_distribution& bernoulli,
                              mt19937_64& prng) {
  // allocate L |S|-element {+1,-1} random vectors
  for (uint32_t i = 0; i < L; i++) {
    random_vectors[i].resize(rvsize);
    for (uint32_t j = 0; j < rvsize; j++) {
      random_vectors[i][j] = 2 * static_cast<int>(bernoulli(prng)) - 1;
    }
  }

#ifdef VERBOSE 
  cout << "Random vectors:\n";
  for (uint32_t i = 0; i < L; i++) {
    cout << "\t";
    for (int rv_i : random_vectors[i]) {
      cout << rv_i << " ";
    }
    cout << endl;
  }
#endif
}

void construct_simhash_sketches(const vector<shingle_vector>& shingle_vectors,
                                const vector<vector<int>>& random_vectors,
                                vector<bitset<L>>& simhash_sketches) {
  // compute SimHash sketches
  for (uint32_t i = 0; i < simhash_sketches.size(); i++) {
    construct_simhash_sketch(simhash_sketches[i], shingle_vectors[i],
                             random_vectors);
  }

#ifdef DEBUG
  cout << "SimHash sketches:\n";
  for (uint32_t i = 0; i < simhash_sketches.size(); i++) {
    cout << "\t" << simhash_sketches[i].to_string() << endl;
  }
#endif
}

void perform_lsh_banding(const vector<uint32_t>& normal_gids,
                         const vector<bitset<L>>& simhash_sketches,
                         vector<unordered_map<bitset<R>,vector<uint32_t>>>&
                            hash_tables) {
  // LSH-banding: assign graphs to hashtable buckets
  for (auto& gid : normal_gids) {
    hash_bands(gid, simhash_sketches[gid], hash_tables);
  }
#ifdef DEBUG
  cout << "Hash tables after hashing bands:\n";
  for (uint32_t i = 0; i < B; i++) {
    cout << "\tHash table " << i << ":\n";
    for (auto& kv : hash_tables[i]) {
      // print graph id's in this bucket
      cout << "\t\tBucket => ";
      for (uint32_t j = 0; j < kv.second.size(); j++) {
        cout << kv.second[j] << " ";
      }
      cout << endl;
    }
  }
#endif
}

void print_lsh_clusters(const vector<uint32_t>& normal_gids,
                        const vector<bitset<L>>& simhash_sketches,
                        const vector<unordered_map<bitset<R>,vector<uint32_t>>>&
                            hash_tables) {
  unordered_set<uint32_t> graphs(normal_gids.size());
  for (auto& gid : normal_gids) {
    graphs.insert(gid);
  }

  while (!graphs.empty()) {
    uint32_t gid = *(graphs.begin());
    unordered_set<uint32_t> cluster;

    queue<uint32_t> q;
    q.push(gid);
    while (!q.empty()) {
      uint32_t g = q.front();
      q.pop();

      cluster.insert(g);

      unordered_set<uint32_t> shared_bucket_graphs;
      get_shared_bucket_graphs(simhash_sketches[g], hash_tables,
                               shared_bucket_graphs);

#ifdef DEBUG
      cout << "\tGraphs sharing buckets with: " << g << " => ";
      for (auto& e : shared_bucket_graphs) {
       cout << e << " ";
      }
      cout << endl;
#endif

      for (auto& h : shared_bucket_graphs) {
        if (cluster.find(h) == cluster.end()) {
          q.push(h);
        }
      }
    }

    for (auto& e : cluster) {
      cout << e << " ";
    }
    cout << endl;

    for (auto& e : cluster) {
      graphs.erase(e);
    }
  }
}

void test_anomalies(uint32_t num_graphs,
                    const vector<bitset<L>>& simhash_sketches,
                    const vector<unordered_map<bitset<R>,vector<uint32_t>>>&
                      hash_tables) {
  // for each attack graph, hash it to the B hash tables
  // if any bucket hashed to contains a graph, the attack is not an anomaly
  // otherwise, the graph is isolated, and is an anomaly
  for (uint32_t gid = 0; gid < num_graphs; gid++) {
    cout << gid << "\t";
    if (is_isolated(simhash_sketches[gid], hash_tables)) {
      cout << "T" << endl;
    } else {
      cout << "F" << endl;
    }
  }
}
