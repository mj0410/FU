#include <seqan3/io/sequence_file/all.hpp>
#include <seqan3/search/all.hpp>
#include <seqan3/core/debug_stream.hpp>       // pretty printing
#include <seqan3/search/algorithm/search.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
using namespace std;
using namespace seqan3;
#include "KmerGen.h"

string FastA_Transform(istream& input)
{

    string line, name, content;
    while( getline( input, line ).good() )
    {
        if( line.empty() || line[0] == '>' )  // Identifier marker
        {
            if( !name.empty() )  // Print out what we read from the last entry
            {
                // std::cout << name << " : " << content << std::endl;
                name.clear();
            }
            if( !line.empty() )
            {
                name = line.substr(1);
            }
            content.clear();
        }
        else if( !name.empty() )
        {
            if( line.find(' ') != std::string::npos )  // Invalid sequence--no spaces allowed
            {
                name.clear();
                content.clear();
            }
            else
            {
                content += line;
            }
        }
    }
    /*if( !name.empty() ){ // Print out what we read from the last entry
        std::cout << name << " : " << content << std::endl;
    }*/
    return content;
}

int main()
{
    ifstream reference_mouse_Y_file ("chr21_dog.fa");
    const clock_t kmer_begin_time = clock();
    std::string s2_seq{ FastA_Transform(reference_mouse_Y_file)};
    kmer s2_k4_mer(s2_seq, 10);
    cout <<" Building k-mer-10: "<< float( clock () - kmer_begin_time ) /  CLOCKS_PER_SEC << '\n';
    std::cout << "Found size: " << s2_k4_mer.size() << " mers\n";

    //Deduplicate
    s2_k4_mer.deDuplicate();

    std::cout << "Total Remaining:  " << s2_k4_mer.size() << " mers\n";
    // write-to-file
    ofstream file;    //Create file pointer variable
    file.open("queries.10.fa");
    int bullet = 1;
    for(auto &i : s2_k4_mer.value())
    {
        file << ">q_" << bullet << "\n";
        file  << i << "\n";
        bullet++;
    }
    file.close();

    sequence_file_input reference_in{"chr21_h.fa"}; // Read the reference (only 1 contig).
    sequence_file_input query_in{"queries.10.fa", fields<field::SEQ>{}}; // Read the query file (multiple contigs).

    const clock_t begin_time = clock();
    fm_index index{get<field::SEQ>(*reference_in.begin())}; // Create an FM-index from the reference.
    cout <<" Building Index Time: "<< float( clock () - begin_time ) /  CLOCKS_PER_SEC << '\n';

    seqan3::configuration const cfg =     seqan3::search_cfg::parallel{4} |
    seqan3::search_cfg::mode{ seqan3::search_cfg::best};

    const clock_t search_begin_time = clock();
    size_t counter{0u};
    for (auto & [seq] : query_in) // For each read in our query file.
        counter += std::ranges::size(search(seq, index,cfg)); // Count if there are more than 100 occurrences.
    std::cout << "Found " << counter << "\n";
    cout <<" Search Time: "<< float( clock () - search_begin_time ) /  CLOCKS_PER_SEC << '\n';


}
