#include "helper.h"

// Returns <vertices, edges>
pair<ll, ll> parboil_take_input(char *filename, map<ll, vector<pair<ll, ll> > > &vertex_list) 
{
    pair<ll, ll> vert_edge;
    ifstream in(filename);
    // Input vertices and edges
    in >> vert_edge.first >> vert_edge.second;
    // No idea what this is
    ll rando;
    in >> rando;
    vector<ll> edge_arr(vert_edge.first);
    ll count = 0;
    while(count < vert_edge.first) {
        // First input is start index of edges, second input is number of edges
        // First input is not needed to construct vertex list
        in >> rando >> edge_arr[count];
        ++count;
    }
    count = 0;
    while(!in.eof()) {
        while(edge_arr[count] > 0) {
            ll u, w;
            in >> u >> w;
            vertex_list[count].push_back(pair<ll, ll>(u, w));
            --edge_arr[count];
        }
        ++count;
    }
    
    in.close();
    return vert_edge;
}

void parboil_create_output(char *filename, map<ll, vector<pair<ll, ll> > > &vertex_list, pair<ll, ll> vert_edge) 
{
    ofstream out(filename);
    printf("%s\n", filename);
    out << vert_edge.first << " " << vert_edge.second << " 0\n";
    int count = 0;
    for(int i = 0; i < vert_edge.first; ++i) {
        out << count << " " << vertex_list[i].size() << "\n";
        count += vertex_list[i].size();
    }
    out << "\n";
    for(int i = 0; i < vert_edge.first; ++i) {
        for(auto j = vertex_list[i].begin(); j != vertex_list[i].end(); ++j) {
            out << j->first << " " << j->second << "\n";
        }
    }
    out.close();
}
