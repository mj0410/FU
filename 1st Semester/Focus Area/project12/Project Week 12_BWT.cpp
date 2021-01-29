#include <bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <ctime>
using namespace std;
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include "Bitvector.h"

std::string bwt_construction(std::string const & text)
{
    std::string bwt{};
    std::vector<uint64_t> sa(text.size() + 1);
    std::iota(sa.begin(), sa.end(), 0);
    std::sort(sa.begin(), sa.end(), [&text](uint64_t a, uint64_t b) -> bool
    {
        while (a < text.size() && b < text.size() && text[a] == text[b])
        {
            ++a, ++b;
        }
        if (b == text.size())
            return false;
        if (a == text.size())
            return true;
        return text[a] < text[b];
    });
    for (auto x : sa)
    {
        if (!x)
            bwt += "$";
        else
            bwt += text[x-1];
    }
    return bwt;
}
size_t to_index(char const chr)
{
    switch (chr)
    {
        case '$': return 0;
        case 'A': return 1;
        case 'C': return 2;
        case 'G': return 3;
        case 'T': return 4;
        case 'N': return 5;
        default: throw std::logic_error{"There is an illegal character in your text."};
    }
}
std::vector<uint16_t> compute_count_table(std::string const & bwt)
{
    std::vector<uint16_t> count_table(6);
    for (auto chr : bwt)
    {
        for (size_t i = to_index(chr) + 1; i < 6; ++i)
            ++count_table[i];
    }
    return count_table;
}
struct occurrence_table
{
    // The list of bitvectors:
    std::vector<Bitvector> data;
    // We want that 5 bitvectors are filled depending on the bwt,
    // so let's customise the constructor of occurrence_table:
    occurrence_table(string const & bwt)
    {
       // cout << "occurrence_table" << bwt.size();
        // resize the 5 bitvectors to the length of the bwt:
        data.resize(6, Bitvector((bwt.size())));
        //cout << "resize the 5 bitvectors to the length of the bwt";
        // fill the bitvectors

        for (size_t i = 0; i < bwt.size(); i++)
            data[to_index(bwt[i])].write(i, 1);
             // cout << "fill the bitvectors 1";
        for (Bitvector & bitv : data)
            bitv.construct(3, 6);
            // cout << "fill the bitvectors 2";
    }


    size_t read(char const chr, size_t const i) const
    {
        return data[to_index(chr)].rank(i + 1);
    }
};


bool contains_number(const string &c)
{
    return (c.find_first_of("0123456789") != string::npos);
}


size_t count(std::string const & P,
             std::string const & bwt,
             std::vector<uint16_t> const & C,
             occurrence_table const & Occ)
{
    int64_t i = P.size() - 1;
    size_t a = 0;
    size_t b = bwt.size() - 1;
    while ((a <= b) && (i >= 0))
    {
        char c = P[i];
        a = C[to_index(c)] + (a ? Occ.read(c, a - 1) : 0);
        b = C[to_index(c)] + Occ.read(c, b) - 1;
        i = i - 1;
    }
    if (b < a)
        return 0;
    else
        return (b - a + 1);
}
int main()
{
      string reference_chr4;
      //string pattern = "TAAAA"; //ATG
string pattern ;
      ifstream reference_chr4_file ("chr4.txt");
      ifstream myfile("queries.fasta");
      if (reference_chr4_file.is_open())
      {
        getline (reference_chr4_file,reference_chr4);
        //cout << reference_chr4 << '\n';
        reference_chr4_file.close();
      }else{
        cout << "Unable to open reference_chr4_file";
      }
      // 10.000,
      // 100.000,
      // 500.000,
      // 1.000.000

    //std::string text{reference_chr4.substr (0,1000000)};
    std::string text{reference_chr4 };
    const clock_t begin_time = clock();
    // compute the bwt, C and Occ:
    std::string bwt = bwt_construction(text);
    // count table
    // -------------------------------------------------------------------------
    std::vector<uint16_t> count_table = compute_count_table(bwt);
    std::cout << "$ A T C G N \n" << "---------\n";
    for (auto c : count_table)
        std::cout << c << " ";
    std::cout << '\n';

    // occurrence table
    // -------------------------------------------------------------------------
    //cout << " occurrence table ";
    occurrence_table Occ(bwt); // construction with bwt as the parameter
    cout <<" Building Index Time: "<< float( clock () - begin_time ) /  CLOCKS_PER_SEC << '\n';
    clock_t search_begin_time = clock();

    int index = 0;
   while (!myfile.eof() && index<=1000000) // <---- change limit size here
        {
            getline(myfile, pattern);

            if (contains_number(pattern))
            {
                index += 1;

            }
            else {
                    cout << "simulated."<< index << ": " << pattern << "\n";

                    std::cout << "# Found: " << count(pattern, bwt, count_table, Occ) << '\n';

            }
        }
    cout <<" Search Time for #1000000 : "<< float( clock () - search_begin_time ) /  CLOCKS_PER_SEC << '\n';
      myfile.close();

  return 0;
}
