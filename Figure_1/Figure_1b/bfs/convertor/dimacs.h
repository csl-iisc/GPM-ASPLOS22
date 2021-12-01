#include "helper.h"

// Returns <vertices, edges>
pair<ll, ll> dimacs_take_input(char *filename, map<ll, vector<pair<ll, ll> > > &vertex_list) 
{
    pair<ll, ll> vert_edge;
    ifstream in(filename);
    while(!in.eof()) {
        char c;
        in >> c;
        switch(c) {
            case 'c': // Comment, skip entire line
                in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            break;
            case 'p':
            {
                string s;
                in >> s; // Should be characters 'sp'
                in >> vert_edge.first >> vert_edge.second;
            }
            break;
            case 'a':
            {
                ll v1, v2, w;
                in >> v1 >> v2 >> w;
                // Convert from 1-indexed to 0-indexed
                --v1, --v2;
                // Insert into map
                vertex_list[v1].push_back(pair<ll, ll>(v2, w));
            }
            break;
        }
    }
    in.close();
    return vert_edge;
}

void dimacs_create_output(char *filename, map<ll, vector<pair<ll, ll> > > &vertex_list, pair<ll, ll> vert_edge) 
{
    ofstream out(filename);
    out << "p sp " << vert_edge.first << " " << vert_edge.second << "\n";
    for(auto i = vertex_list.begin(); i != vertex_list.end(); ++i) {
        for(auto j = i->second.begin(); j != i->second.end(); ++j) {
            out << "a " << i->first + 1 << " " << j->first + 1 << " " << j->second << "\n";
        }
    }
    out.close();
}
