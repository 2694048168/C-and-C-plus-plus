/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief The shortest path between nodes
 * @version 0.1
 * @date 2024-01-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

/**
 * @brief The shortest path between nodes
 * 
 * Write a program that, given a network of nodes and the distances between them, 
 * computes and displays the shortest distance from a specified node to all the others,
 * as well as the path between the start and end node.
 * As input, consider the following undirected graph
 * 
 * To solve the proposed problem you must use the Dijkstra algorithm for finding
 * the shortest path in a graph. Although the original algorithm finds the shortest path
 * between two given nodes, the requirement here is to find the shortest path between
 * one specified node and all the others in the graph, 
 * which is another version of the algorithm.
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
template<typename Vertex = int, typename Weight = double>
class graph
{
public:
    typedef Vertex                     vertex_type;
    typedef Weight                     weight_type;
    typedef std::pair<Vertex, Weight>  neighbor_type;
    typedef std::vector<neighbor_type> neighbor_list_type;

public:
    void add_edge(const Vertex source, const Vertex target, const Weight weight, const bool bidirectional = true)
    {
        adjacency_list[source].push_back(std::make_pair(target, weight));
        adjacency_list[target].push_back(std::make_pair(source, weight));
    }

    size_t vertex_count() const
    {
        return adjacency_list.size();
    }

    std::vector<Vertex> verteces() const
    {
        std::vector<Vertex> keys;
        for (const auto &kvp : adjacency_list) keys.push_back(kvp.first);
        return keys;
    }

    const neighbor_list_type &neighbors(const Vertex &v) const
    {
        auto pos = adjacency_list.find(v);
        if (pos == adjacency_list.end())
            throw std::runtime_error("vertex not found");

        return pos->second;
    }

    constexpr static Weight Infinity = std::numeric_limits<Weight>::infinity();

private:
    std::map<vertex_type, neighbor_list_type> adjacency_list;
};

template<typename Vertex, typename Weight>
void shortest_path(const graph<Vertex, Weight> &g, const Vertex source, std::map<Vertex, Weight> &min_distance,
                   std::map<Vertex, Vertex> &previous)
{
    const auto n        = g.vertex_count();
    const auto verteces = g.verteces();

    min_distance.clear();
    for (const auto &v : verteces) min_distance[v] = graph<Vertex, Weight>::Infinity;
    min_distance[source] = 0;

    previous.clear();

    std::set<std::pair<Weight, Vertex>> vertex_queue;
    vertex_queue.insert(std::make_pair(min_distance[source], source));

    while (!vertex_queue.empty())
    {
        auto dist = vertex_queue.begin()->first;
        auto u    = vertex_queue.begin()->second;

        vertex_queue.erase(std::begin(vertex_queue));

        const auto &neighbors = g.neighbors(u);
        for (const auto &neighbor : neighbors)
        {
            auto v          = neighbor.first;
            auto w          = neighbor.second;
            auto dist_via_u = dist + w;
            if (dist_via_u < min_distance[v])
            {
                vertex_queue.erase(std::make_pair(min_distance[v], v));

                min_distance[v] = dist_via_u;
                previous[v]     = u;
                vertex_queue.insert(std::make_pair(min_distance[v], v));
            }
        }
    }
}

template<typename Vertex>
void build_path(const std::map<Vertex, Vertex> &prev, const Vertex v, std::vector<Vertex> &result)
{
    result.push_back(v);

    auto pos = prev.find(v);
    if (pos == std::end(prev))
        return;

    build_path(prev, pos->second, result);
}

template<typename Vertex>
std::vector<Vertex> build_path(const std::map<Vertex, Vertex> &prev, const Vertex v)
{
    std::vector<Vertex> result;
    build_path(prev, v, result);
    std::reverse(std::begin(result), std::end(result));
    return result;
}

template<typename Vertex>
void print_path(const std::vector<Vertex> &path)
{
    for (size_t i = 0; i < path.size(); ++i)
    {
        std::cout << path[i];
        if (i < path.size() - 1)
            std::cout << " -> ";
    }
}

graph<char, double> make_graph()
{
    graph<char, double> g;
    g.add_edge('A', 'B', 4);
    g.add_edge('A', 'H', 8);
    g.add_edge('B', 'C', 8);
    g.add_edge('B', 'H', 11);
    g.add_edge('C', 'D', 7);
    g.add_edge('C', 'F', 4);
    g.add_edge('C', 'J', 2);
    g.add_edge('D', 'E', 9);
    g.add_edge('D', 'F', 14);
    g.add_edge('E', 'F', 10);
    g.add_edge('F', 'G', 2);
    g.add_edge('G', 'J', 6);
    g.add_edge('G', 'H', 1);
    g.add_edge('H', 'J', 7);

    return g;
}

graph<char, double> make_graph_wiki()
{
    graph<char, double> g;
    g.add_edge('A', 'B', 7);
    g.add_edge('A', 'C', 9);
    g.add_edge('A', 'F', 14);
    g.add_edge('B', 'C', 10);
    g.add_edge('B', 'D', 15);
    g.add_edge('C', 'D', 11);
    g.add_edge('C', 'F', 2);
    g.add_edge('D', 'E', 6);
    g.add_edge('E', 'F', 9);

    return g;
}

graph<std::string, double> make_graph_map()
{
    graph<std::string, double> g;

    g.add_edge("London", "Reading", 41);
    g.add_edge("London", "Oxford", 57);
    g.add_edge("Reading", "Swindon", 40);
    g.add_edge("Swindon", "Bristol", 40);
    g.add_edge("Oxford", "Swindon", 30);
    g.add_edge("London", "Southampton", 80);
    g.add_edge("Southampton", "Bournemouth", 33);
    g.add_edge("Bournemouth", "Exeter", 89);
    g.add_edge("Bristol", "Exeter", 83);
    g.add_edge("Bristol", "Bath", 12);
    g.add_edge("Swindon", "Bath", 35);
    g.add_edge("Reading", "Southampton", 50);

    return g;
}

// ------------------------------
int main(int argc, char **argv)
{
    {
        auto g = make_graph();

        char                   source = 'A';
        std::map<char, double> min_distance;
        std::map<char, char>   previous;
        shortest_path(g, source, min_distance, previous);

        for (const auto &kvp : min_distance)
        {
            std::cout << source << " -> " << kvp.first << " : " << kvp.second << '\t';

            print_path(build_path(previous, kvp.first));

            std::cout << std::endl;
        }
    }

    {
        auto g = make_graph_wiki();

        char                   source = 'A';
        std::map<char, double> min_distance;
        std::map<char, char>   previous;
        shortest_path(g, source, min_distance, previous);

        for (const auto &kvp : min_distance)
        {
            std::cout << source << " -> " << kvp.first << " : " << kvp.second << '\t';

            print_path(build_path(previous, kvp.first));

            std::cout << std::endl;
        }
    }

    {
        auto g = make_graph_map();

        std::string                        source = "London";
        std::map<std::string, double>      min_distance;
        std::map<std::string, std::string> previous;
        shortest_path(g, source, min_distance, previous);

        for (const auto &kvp : min_distance)
        {
            std::cout << source << " -> " << kvp.first << " : " << kvp.second << '\t';

            print_path(build_path(previous, kvp.first));

            std::cout << std::endl;
        }
    }

    return 0;
}
