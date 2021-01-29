# include <bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <ctime>

using namespace std;

int ASIZE = 256;
void preBmBc(char *x, int m, int bmBc[]) {
   int i;

   for (i = 0; i < ASIZE; ++i)
      bmBc[i] = m;

   for (i = 0; i < m - 1; ++i)
      bmBc[x[i]] = m - i - 1;
}

void HORSPOOL(char *pattern , int patter_length, char *text, int text_length) {
   int j, bmBc[ASIZE];
   char c;

   /* Preprocessing */
   preBmBc(pattern, patter_length, bmBc);

   /* Searching */
   j = 0;
   while (j <= text_length - patter_length) {
      c = text[j + patter_length - 1];
      if (pattern[patter_length - 1] == c && memcmp(pattern, text + j, patter_length - 1) == 0)
           cout << "Found at position:" << j << '\n';
      j += bmBc[c];
   }
}

bool contains_number(const string &c)
{
    return (c.find_first_of("0123456789") != string::npos);
}


int main ()
{
  string reference_chr4;
  string pattern = "TAAAA";
  //string pattern = "AGGGTAAAA";

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

   //string str = reference_chr4;//.substr (0,10001);

  // string str = reference_chr4.substr (0,100001);

  // string str = reference_chr4.substr (0,500001);

  // string str = reference_chr4.substr (0,1000001);
   // string str = "The quick brown fox jumps over the lazy dog";
   // string pattern = "fox";
    char cstr[reference_chr4.size()+1];
	strcpy(cstr, reference_chr4.c_str());	// or pass &s[0]
    int count = 0;
    const clock_t begin_time = clock();
     while (!myfile.eof() && count<=1000000) // <---- change limit size here
        {
            getline(myfile, pattern);

            if (contains_number(pattern))
            {
                count += 1;

            }
            else {
                    cout << "simulated."<< count << ": " << pattern << "\n";
                    char cstr_pattern[pattern.size()+1];
                    strcpy(cstr_pattern, pattern.c_str());
                    HORSPOOL(cstr_pattern , pattern.size() ,cstr ,reference_chr4.size());
            }
        }
    //HORSPOOL(cstr_pattern , pattern.size() ,cstr ,str.size());
      myfile.close();
    cout <<" Search Time: "<< float( clock () - begin_time ) /  CLOCKS_PER_SEC;
  return 0;
}

