#include <cstdio>
#include <cstring>
#include <algorithm>
#include <iostream>
using namespace std;
#define MAXN 65536
#define MAXLG 17

char A[MAXN];

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
    int k, ret = 0;
    if (x == y) return N - x;
    for (k = stp - 1; k >= 0 && x < N && y < N; k --) {
            if (P[k][x] == P[k][y]) {
                    x += 1 << k, y += 1 << k, ret += 1 << k;
            }
    }
    return ret;
}

int main(void)
{
    gets(A);
    cout << "A : " << A << endl;
    for (N = strlen(A), i = 0; i < N; i ++)
        P[0][i] = A[i] - 'a';
    for (stp = 1, cnt = 1; cnt >> 1 < N; stp ++, cnt <<= 1) {
            for (i = 0; i < N; i ++)
            {
                L[i].nr[0] = P[stp - 1][i];
                L[i].nr[1] = i + cnt < N ? P[stp - 1][i + cnt] : -1;
                L[i].p = i;
            }
            sort(L, L + N, cmp);
            for (i = 0; i < N; i ++)
                P[stp][L[i].p] = i > 0 && L[i].nr[0] == L[i - 1].nr[0] && L[i].nr[1] == L[i - 1].nr[1] ? P[stp][L[i - 1].p] : i;
    }
    cout << "\n";

    cout << "Suffix Array : ";
    for (int a = 0; a < strlen(A); a++) {
        cout << P[stp-1][a] << " ";
    }
    cout << "\n";
    cout << "\n";

    for (int a = 0; a < strlen(A); a++) {
            for (int b = a; b < strlen(A); b++) {
                    cout << A[b];
            }
            for (int c = 0; c < a; c++) {
                    cout << A[c];
            }
            //cout << " " << P[stp-1][a] << "\n";
            cout << "\n";
    }

    int m = 0;
    for (int num = 0; num < strlen(A); num++){
        if (P[stp-1][num] == 0) {
            m = num;
        }
    }

    cout << "\n";
    cout << "The smallest among them is ";

    for (int b = m; b < strlen(A); b++) {
            cout << A[b];
    }
    for (int c = 0; c < m; c++) {
            cout << A[c];
    }
        cout << "\n";

    return 0;
}
