#ifndef KMERGEN_H_INCLUDED
#define KMERGEN_H_INCLUDED


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
