#include <cstdio>
#include <cstring>
#include <algorithm>
#include <iostream>
#include<bits/stdc++.h>
using namespace std;
#define MAXN 65536
#define MAXLG 17

char A[MAXN];
char B[MAXN];
char M[MAXN];
int sa[MAXN];

struct entry {
    int nr[2], p;
} L[MAXN];

int P[MAXLG][MAXN], N, i, stp, cnt;

int cmp(struct entry a, struct entry b)
{
    return a.nr[0] == b.nr[0] ? (a.nr[1] < b.nr[1] ? 1 : 0) : (a.nr[0] < b.nr[0] ? 1 : 0);
}

int lcp(int x, int y)
{
    x = sa[x];
    y = sa[y];
    int k, ret = 0;
    if (x == y) return N - x;
    for (k = stp - 1; k >= 0 && x < N && y < N; k --) {
            //cout << P[k][x] << ", " << P[k][y] << endl;
            if (P[k][x] == P[k][y]) {
                    x += 1 << k, y += 1 << k, ret += 1 << k;
            }
    }
    return ret;
}

int main(void)
{
    gets(A);

    for (int i = 0; i < strlen(A); i++) {
        B[i] = A[strlen(A)-i-1];
    }

    //cout << "A : " << A << endl;
    //cout << "B : " << B << endl;
    //cout << "strlen(A) : " << strlen(A) << endl;

    for (int j = 0; j < 2*strlen(A)+1; j++){
        if (j < strlen(A)) M[j] = A[j];
        else if (j == strlen(A)) M[j] = '#';
        else M[j] = B[j-strlen(A)-1];
    }

    //cout << "Merged : " << M << endl;

    for (N = strlen(M), i = 0; i < N; i ++)
        P[0][i] = M[i] - 'a';
    for (stp = 1, cnt = 1; cnt >> 1 < N; stp ++, cnt <<= 1) {
            for (i = 0; i < N; i ++)
            {
                L[i].nr[0] = P[stp - 1][i];
                L[i].nr[1] = i + cnt < N ? P[stp - 1][i + cnt] : -1;
                L[i].p = i;
                //cout << "L[i].nr[0] : " << L[i].nr[0] << ", L[i].nr[1] : " << L[i].nr[1] << ", L[i].p : " << L[i].p << endl;
            }
            sort(L, L + N, cmp);
            for (i = 0; i < N; i ++)
                P[stp][L[i].p] = i > 0 && L[i].nr[0] == L[i - 1].nr[0] && L[i].nr[1] == L[i - 1].nr[1] ? P[stp][L[i - 1].p] : i;
    }

    /*
    for (int k = 0; k < strlen(M); k++) {
            for (int j = 0; j < strlen(M); j++) {
                cout << P[k][j] << " ";
            }
            cout << "\n";
    }
    */

    for (int k = 0; k < strlen(M); k++) {
        sa[P[stp-1][k]] = k;
    }

    cout << "suffix array : " ;
    for (int i = 0; i<strlen(M); i++) cout << sa[i] << " ";
    cout << "\n";


    /*
    for (int a = 0; a < strlen(M); a++) {
            for (int b = a; b < strlen(M); b++) {
                    cout << M[b];
            }
            cout << "\n";
    }
    */

    int cp[strlen(M)];
    //int test = lcp(3,9);
    //cout << test;

    cout << "lcp : " ;
    for (int x = 0; x < strlen(M); x++) {
        if (x == 0) cp[x] = 0;
        else cp[x] = lcp(x-1, x);

        cout << cp[x] << " ";
    }
    cout << "\n";

    int len = 0;
    int po = 0;
    for (int i = 1; i < strlen(M); i++) {
        if(cp[i] > len) {
            if((sa[i-1] < strlen(A) && sa[i] > strlen(A)) || (sa[i] < strlen(A) && sa[i-1] > strlen(A))) {
               len = cp[i];
               po = sa[i];
               //cout << "i : " << i << endl;
               //cout << len << ", " << po << endl;
            }
        }
    }

    cout << "longest palindrome is ";
    for (int i = po; i < po+len; i++) {
        cout << M[i];
    }
    cout << "\n";

    return 0;

}
