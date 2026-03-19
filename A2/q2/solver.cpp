#include <iostream>
#include <vector>
#include <unordered_set>
#include <string>
#include <fstream>
#include <random>
#include <algorithm>
#include <queue>
#include <set>
#include <chrono>

using namespace std;


int TIME1_MINS = 5; 
int TIME2_MINS = 10; 
int TIME3_MINS = 10;

struct Edge {
    int u, v;
    double p;
};

typedef vector<vector<int>> Adj;

void write_output(const vector<int>& blocked_indices,
                  const vector<Edge>& edges,
                  const string& path,
                  int K) {
    ofstream out(path, ios::out | ios::trunc);

    unordered_set<int> used;
    int written = 0;

    for (int idx : blocked_indices) {
        if (idx < 0 || idx >= (int)edges.size()) continue;
        if (used.count(idx)) continue;
        out << edges[idx].u << " " << edges[idx].v << "\n";
        used.insert(idx);
        written++;
        if (written == K) break;
    }

    for (int i = 0; i < (int)edges.size() && written < K; ++i) {
        if (!used.count(i)) {
            out << edges[i].u << " " << edges[i].v << "\n";
            used.insert(i);
            written++;
        }
    }

    out.flush();
    out.close();
}

vector<int> build_output_edges(const vector<int>& chosen, const vector<int>& fallback, int K) {
    vector<int> result;
    unordered_set<int> used;

    for (int idx : chosen) {
        if (!used.count(idx)) {
            result.push_back(idx);
            used.insert(idx);
            if ((int)result.size() == K) return result;
        }
    }

    for (int idx : fallback) {
        if (!used.count(idx)) {
            result.push_back(idx);
            used.insert(idx);
            if ((int)result.size() == K) return result;
        }
    }

    return result;
}

vector<int> get_min_hops(int n, const Adj& adj, const vector<int>& seeds) {
    vector<int> dist(n, -1);
    queue<int> q;

    for (int s : seeds) {
        if (s >= 0 && s < n && dist[s] == -1) {
            dist[s] = 0;
            q.push(s);
        }
    }

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v : adj[u]) {
            if (dist[v] == -1) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }

    return dist;
}

long long evaluate_blocked_set(int num_sim, int n, const vector<int>& seeds, int hops, const vector<vector<vector<int>>>& scenarios, const vector<bool>& is_blocked, const vector<Edge>& edges) {
    long long total_burned = 0;
    vector<int> dist(n, -1);
    vector<int> q;
    q.reserve(n);

    for (int s = 0; s < num_sim; ++s) {
        fill(dist.begin(), dist.end(), -1);
        q.clear();

        int head = 0;
        int burned = 0;

        for (int seed : seeds) {
            if (seed >= 0 && seed < n && dist[seed] == -1) {
                dist[seed] = 0;
                q.push_back(seed);
                burned++;
            }
        }

        while (head < (int)q.size()) {
            int u = q[head++];

            if (hops != -1 && dist[u] >= hops) continue;

            for (int e_idx : scenarios[s][u]) {
                if (is_blocked[e_idx]) continue;

                int v = edges[e_idx].v;
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    burned++;
                    q.push_back(v);
                }
            }
        }

        total_burned += burned;
    }

    return total_burned;
}

int main(int argc, char* argv[]) {
    if (argc < 7) return 1;

    string graph_path = argv[1];
    string seed_path = argv[2];
    string output_path = argv[3];
    int K = stoi(argv[4]);
    int num_sim = stoi(argv[5]);
    int hops = stoi(argv[6]);

    auto start_time = chrono::steady_clock::now();
    auto elapsed_secs = [&]() {
        return chrono::duration_cast<chrono::seconds>(
            chrono::steady_clock::now() - start_time
        ).count();
    };

    mt19937 rng(1337);
    uniform_real_distribution<double> dist_roll(0.0, 1.0);

    cout << "[" << elapsed_secs() << "] Loading seeds...\n";
    vector<int> seeds;
    vector<bool> is_seed;
    ifstream sf(seed_path);

    int s_node;
    int max_seed = -1;
    while (sf >> s_node) {
        seeds.push_back(s_node);
        max_seed = max(max_seed, s_node);
    }
    sf.close();

    cout << "[" << elapsed_secs() << "] Loading graph...\n";
    vector<Edge> edges;
    int max_id = max_seed;
    ifstream gf(graph_path);

    int u, v;
    double p;
    while (gf >> u >> v >> p) {
        edges.push_back({u, v, p});
        max_id = max(max_id, max(u, v));
    }
    gf.close();

    int n = max_id + 1;

    is_seed.assign(n, false);
    for (int s : seeds) {
        if (s >= 0 && s < n) is_seed[s] = true;
    }

    cout << "[" << elapsed_secs() << "] Pruning candidates...\n";
    vector<int> valid_candidates;

    if (hops != -1) {
        Adj full_adj(n);
        for (const auto& e : edges) {
            full_adj[e.u].push_back(e.v);
        }

        vector<int> min_dist = get_min_hops(n, full_adj, seeds);

        for (int i = 0; i < (int)edges.size(); ++i) {
            int src = edges[i].u;
            if (min_dist[src] != -1 && min_dist[src] < hops) {
                valid_candidates.push_back(i);
            }
        }

        cout << "Pruned from " << edges.size() << " to " << valid_candidates.size() << " candidates based on hops < " << hops << "\n";
    
    } else {
        for (int i = 0; i < (int)edges.size(); ++i) {
            valid_candidates.push_back(i);
        }
        cout << "No pruning applied, all " << edges.size()
             << " edges are candidates.\n";
    }

    cout << "[" << elapsed_secs() << "] Ordering candidates...\n";
    vector<int> a0_edges;
    vector<int> non_a0_edges;

    for (int idx : valid_candidates) {
        if (is_seed[edges[idx].u]) a0_edges.push_back(idx);
        else non_a0_edges.push_back(idx);
    }

    sort(a0_edges.begin(), a0_edges.end(), [&](int a, int b) {
        return edges[a].p > edges[b].p;
    });

    vector<int> initial_blocked;
    for (int idx : a0_edges) {
        if ((int)initial_blocked.size() == K) break;
        initial_blocked.push_back(idx);
    }

    write_output(initial_blocked, edges, output_path, K);

    vector<int> ordered_candidates;
    ordered_candidates.reserve(valid_candidates.size());

    for (int idx : a0_edges) ordered_candidates.push_back(idx);
    for (int idx : non_a0_edges) ordered_candidates.push_back(idx);

    int MAX_CANDIDATES = 100000;
    if ((int)ordered_candidates.size() > MAX_CANDIDATES) {
        ordered_candidates.resize(MAX_CANDIDATES);
    }

    cout << "Using " << ordered_candidates.size() << " candidates for optimization\n";

    cout << "[" << elapsed_secs() << "] Generating scenarios...\n";
    vector<vector<vector<int>>> scenarios(num_sim, vector<vector<int>>(n));

    for (int i = 0; i < num_sim; ++i) {
        for (int e_idx = 0; e_idx < (int)edges.size(); ++e_idx) {
            if (dist_roll(rng) <= edges[e_idx].p) {
                scenarios[i][edges[e_idx].u].push_back(e_idx);
            }
        }
    }

    vector<int> current_blocked;
    vector<bool> is_blocked(edges.size(), false);

    cout << "[" << elapsed_secs() << "] Phase 1: Greedy marginal selection...\n";

    long long phase1_score = evaluate_blocked_set(
        num_sim, n, seeds, hops, scenarios, is_blocked, edges
    );

    unordered_set<int> blocked_set;

    while ((int)current_blocked.size() < K) {
        if (elapsed_secs() >= TIME1_MINS * 60) break;
        if (ordered_candidates.empty()) break;

        long long best_gain = -1;
        int best_edge = -1;

        for (int idx : ordered_candidates) {
            if (blocked_set.count(idx)) continue;

            is_blocked[idx] = true;

            long long new_score = evaluate_blocked_set(
                num_sim, n, seeds, hops, scenarios, is_blocked, edges
            );

            long long gain = phase1_score - new_score;

            is_blocked[idx] = false;

            if (gain > best_gain) {
                best_gain = gain;
                best_edge = idx;
            }
        }

        if (best_edge == -1) break;

        current_blocked.push_back(best_edge);
        blocked_set.insert(best_edge);
        is_blocked[best_edge] = true;

        phase1_score = evaluate_blocked_set(
            num_sim, n, seeds, hops, scenarios, is_blocked, edges
        );

        vector<int> output_edges = build_output_edges(current_blocked, initial_blocked, K);
        write_output(output_edges, edges, output_path, K);

        cout << "Phase 1 iter " << current_blocked.size() << " added edge " << best_edge << " gain=" << best_gain << " score=" << phase1_score << "\n";
    }

    cout << "[" << elapsed_secs() << "] Phase 2: Guided Hill Climbing...\n";

    vector<int> phase2_current = current_blocked;
    vector<bool> phase2_blocked = is_blocked;

    long long phase2_score = evaluate_blocked_set(
        num_sim, n, seeds, hops, scenarios, phase2_blocked, edges
    );

    if (!phase2_current.empty() && !ordered_candidates.empty()) {
        while (elapsed_secs() < (TIME1_MINS + TIME2_MINS) * 60) {
            bool improved = false;

            int trials_out_limit = min(10, (int)phase2_current.size());

            for (int trials_out = 0; trials_out < trials_out_limit; ++trials_out) {
                int out_idx = uniform_int_distribution<int>(
                    0, (int)phase2_current.size() - 1
                )(rng);

                int old_edge = phase2_current[out_idx];
                phase2_blocked[old_edge] = false;

                int trials_in_limit = min(50, (int)ordered_candidates.size());

                for (int trials_in = 0; trials_in < trials_in_limit; ++trials_in) {
                    int new_edge = ordered_candidates[
                        uniform_int_distribution<int>(
                            0, (int)ordered_candidates.size() - 1
                        )(rng)
                    ];

                    if (phase2_blocked[new_edge]) continue;

                    phase2_blocked[new_edge] = true;

                    long long new_score = evaluate_blocked_set(
                        num_sim, n, seeds, hops, scenarios, phase2_blocked, edges
                    );

                    if (new_score < phase2_score) {
                        phase2_score = new_score;
                        phase2_current[out_idx] = new_edge;
                        improved = true;

                        vector<int> output_edges = build_output_edges(phase2_current, initial_blocked, K);
                        write_output(output_edges, edges, output_path, K);

                        cout << "Phase 2 improved: swapped out "
                             << old_edge << " for " << new_edge
                             << ", score=" << phase2_score << "\n";
                        break;
                    }

                    phase2_blocked[new_edge] = false;
                }

                if (improved) break;

                phase2_blocked[old_edge] = true;
            }

        }
    }

    current_blocked = phase2_current;
    is_blocked = phase2_blocked;

    cout << "[" << elapsed_secs() << "] Phase 3: Random swap local search...\n";

    long long phase3_score = evaluate_blocked_set(
        num_sim, n, seeds, hops, scenarios, is_blocked, edges
    );

    vector<int> non_blocked;
    for (int idx : valid_candidates) {
        if (!is_blocked[idx]) non_blocked.push_back(idx);
    }

    if (!current_blocked.empty() && !non_blocked.empty()) {
        while (elapsed_secs() < (TIME1_MINS + TIME2_MINS + TIME3_MINS) * 60) {
            if (current_blocked.empty() || non_blocked.empty()) break;

            int out_idx = uniform_int_distribution<int>(
                0, (int)current_blocked.size() - 1
            )(rng);
            int in_idx = uniform_int_distribution<int>(
                0, (int)non_blocked.size() - 1
            )(rng);

            int old_edge = current_blocked[out_idx];
            int new_edge = non_blocked[in_idx];

            is_blocked[old_edge] = false;
            is_blocked[new_edge] = true;

            long long new_score = evaluate_blocked_set(
                num_sim, n, seeds, hops, scenarios, is_blocked, edges
            );

            if (new_score < phase3_score) {
                cout << "Phase 3 improved: swapped out " << old_edge << " for " << new_edge << " score: " << phase3_score << " -> " << new_score << "\n";

                phase3_score = new_score;
                current_blocked[out_idx] = new_edge;
                non_blocked[in_idx] = old_edge;

                vector<int> output_edges = build_output_edges(current_blocked, initial_blocked, K);
                write_output(output_edges, edges, output_path, K);
            } else {
                is_blocked[old_edge] = true;
                is_blocked[new_edge] = false;
            }
        }
    }

    cout << "[" << elapsed_secs() << "] Finished. Final score: " << phase3_score << "\n";

    return 0;
}