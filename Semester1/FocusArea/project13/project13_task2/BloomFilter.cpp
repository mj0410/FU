#ifndef BLOOM_FILTER_H
#define BLOOM_FILTER_H

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <ctime>
using namespace std;
#include <array>

#include "KmerGen.h"

//#include "MurmurHash3.h"
void MurmurHash3_x64_128(const void *key, int len, uint32_t seed, void *out)
{
    // emulate MurmurHash3_x64_128
    size_t h = std::hash<std::string>()(std::string((const char*)key, len));
    size_t *s = reinterpret_cast<size_t*>(out);
    for (int i = 0; i < 128; i += 8*sizeof(size_t))
        *s++ = h;
}

//basic structure of a bloom filter object
struct BloomFilter {
    BloomFilter(size_t size, uint8_t numHashes);
    void add(const uint8_t *data, std::size_t len);
    bool possiblyContains(const uint8_t *data, std::size_t len) const;
private:
    uint8_t m_numHashes;
    std::vector<bool> m_bits;
};
//Bloom filter constructor
BloomFilter::BloomFilter(size_t size, uint8_t numHashes)
    : m_bits(size),
    m_numHashes(numHashes) {}
//Hash array created using the MurmurHash3 code
static std::array<uint64_t, 2> myhash(const uint8_t *data, std::size_t len)
{
    std::array<uint64_t, 2> hashValue;
    MurmurHash3_x64_128(data, len, 0, hashValue.data());
    return hashValue;
}
//Hash array created using the MurmurHash3 code
inline size_t nthHash(int n,
    uint64_t hashA,
    uint64_t hashB,
    size_t filterSize) {
    return (hashA + n * hashB) % filterSize; // <- not sure if that is OK, perhaps it is.
}
//Adds an element to the array
void BloomFilter::add(const uint8_t *data, std::size_t len) {
    auto hashValues = myhash(data, len);
    for (int n = 0; n < m_numHashes; n++)
    {
        m_bits[nthHash(n, hashValues[0], hashValues[1], m_bits.size())] = true;
    }
}
//Returns true or false based on a probabilistic assesment of the array         using MurmurHash3
bool BloomFilter::possiblyContains(const uint8_t *data, std::size_t   len) const {
    auto hashValues = myhash(data, len);
    for (int n = 0; n < m_numHashes; n++)
    {
        if (!m_bits[nthHash(n, hashValues[0], hashValues[1], m_bits.size())])
        {
            return false;
        }
    }
    return true;
}

string FastA_Transform(istream& input){

  string line, name, content;
    while( getline( input, line ).good() ){
        if( line.empty() || line[0] == '>' ){ // Identifier marker
            if( !name.empty() ){ // Print out what we read from the last entry
               // std::cout << name << " : " << content << std::endl;
                name.clear();
            }
            if( !line.empty() ){
                name = line.substr(1);
            }
            content.clear();
        } else if( !name.empty() ){
            if( line.find(' ') != std::string::npos ){ // Invalid sequence--no spaces allowed
                name.clear();
                content.clear();
            } else {
                content += line;
            }
        }
    }
    /*if( !name.empty() ){ // Print out what we read from the last entry
        std::cout << name << " : " << content << std::endl;
    }*/
 return content;
}
#endif

#include <functional>
#include <iostream>
#include <assert.h>
#include <typeinfo>
#include <math.h>
#include <cmath>

using namespace std;

int main()
{
    ifstream human_21 ("human_chr21.fa");
    ifstream dog_21 ("dog_chr21.fa");

    //pattern
    std::string seq_p{ FastA_Transform(human_21)};
    kmer k4_mer_p(seq_p, 15);
    k4_mer_p.deDuplicate();

    //corpus
    std::string seq_c{FastA_Transform(dog_21)};
    kmer k4_mer_c(seq_c, 15);
    k4_mer_c.deDuplicate();

    std::cout << "Total " << k4_mer_c.size() << " mers in dog chr21\n";
    std::cout << "Total " << k4_mer_p.size() << " mers in human chr21\n";

    int n = k4_mer_p.size();
    float p = 0.1;
    int m = ceil(-(n*log(p)/pow(log(2),2))) + 100;
    int k = floor(m*log(2)/n);
    //int k = 16;

    int num = 0;

    cout << "size of table: " << m << endl;
    cout << "number of hash functions: " << k << endl;

    BloomFilter bf(m, k);
    //k4_mer_p.size(); k4_mer_c.size();

    for (int i = 0; i < k4_mer_p.size(); i++) {
        std::string sp = static_cast<std::string>(k4_mer_p.getByIndex(i));
        bf.add((uint8_t*)sp.c_str(), sp.size());
    }

    for (int i = 0; i < k4_mer_c.size(); i++) {
        std::string sc = static_cast<std::string>(k4_mer_c.getByIndex(i));
        bool possiblyContains  = bf.possiblyContains((uint8_t*)sc.c_str(), sc.size());
        num += possiblyContains;
    }

    cout << num << " are founded" << endl;
}
