/*
#ifndef KMERGEN_H_INCLUDED
#define KMERGEN_H_INCLUDED

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <ctime>

class kmer
{
public:
    kmer();

    kmer(std::string &str, int k) : k_(k), str_(str)
    {
        for(int i = 0; i <= str_.size() - k_; ++i){
            //value_.push_back(str_.substr(i, k_));
            //cout << "substr: " << str_.substr(i, k_) << endl;
            int n = 0;
            for (int j = 0; j < value_.size(); j++){
                if(value_[j]==str_.substr(i, k_)) {
                    cout << "size of table: " << value_.size() << endl;
                    n += 1;
                }
            }
            if (n==0) {
                value_.push_back(str_.substr(i, k_));
            }
        }
    }
    auto value() { return value_; }
    auto get_value_byIndex(int index) { return value_[index]; }
    auto size() { return value_.size(); }

private:
    std::vector<std::string_view> value_;
    std::string_view str_;
    int k_;
};


#endif // KMERGEN_H_INCLUDED
*/

#ifndef KMERGEN_H_INCLUDED
#define KMERGEN_H_INCLUDED

#include <vector>
#include <algorithm>


class kmer
{
public:
    kmer();
    kmer(std::string &str, int k) : k_(k), str_(str)
    {
        for(int i = 0; i <= str_.size() - k_; ++i)
            value_.push_back(str_.substr(i, k_));
    }
    auto getByIndex(int i){
     return value_[i];
    }
    auto deDuplicate(){
        vector<string_view>::iterator ip;

        // Using std::unique
        ip = std::unique(value_.begin(), value_.begin() + value_.size());
        // Now v becomes {1 3 10 1 3 7 8 * * * * *}
        // * means undefined

        // Resizing the vector so as to remove the undefined terms
        value_.resize(std::distance(value_.begin(), ip));
    }
    auto value() { return value_; }
    auto size() { return value_.size(); }
private:
    std::vector<std::string_view> value_;
    std::string_view str_;
    int k_;
};


#endif // KMERGEN_H_INCLUDED
