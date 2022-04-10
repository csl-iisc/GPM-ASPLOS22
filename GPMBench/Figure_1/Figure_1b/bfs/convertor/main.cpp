#include "helper.h"
#include "dimacs.h"
#include "parboil.h"

int main(int argv, char **argc)
{
    if(argv < 3) {
        printf("Format: ./convertor input_file output_file [input_format] [output_format]\n");
        return -1;
    }
    
    // Accept input type
    int input_type = 0, output_type = 1;
    if(argv >= 4) {
        input_type = atoi(argc[3]);
    }
    if(argv >= 5) {
        output_type = atoi(argc[4]);
    }
    
    // Create central hashmap which maps vertex to its 
    // edges, where an edge is a <vertex, weight> pair
    map<ll, vector<pair<ll, ll> > > vertex_list;
    // Number of vertices and edges
    pair<ll, ll> vert_edge;
    
    switch(input_type) {
        case 0:
            vert_edge = dimacs_take_input(argc[1], vertex_list);
            break;
        case 1:
            vert_edge = parboil_take_input(argc[1], vertex_list);
            break;
    }
    
    switch(output_type) {
        case 0:
            dimacs_create_output(argc[2], vertex_list, vert_edge);
            break;
        case 1:
            parboil_create_output(argc[2], vertex_list, vert_edge);
            break;
    }
    
    return 0;
}
