#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>
int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  __m512 x_vec = _mm512_load_ps(x);
  __m512 y_vec = _mm512_load_ps(y);
  __m512 m_vec = _mm512_load_ps(m);
  for(int i=0; i<N; i++) {

    __m512 xi_vec = _mm512_set1_ps(x[i]);
    __m512 yi_vec = _mm512_set1_ps(y[i]);
        

    __mmask16 mask = ~(1 << i); 


    __m512 rx_vec = _mm512_sub_ps(xi_vec, x_vec);
    __m512 ry_vec = _mm512_sub_ps(yi_vec, y_vec);

    __m512 r2_vec = _mm512_add_ps(_mm512_mul_ps(rx_vec, rx_vec), _mm512_mul_ps(ry_vec, ry_vec));


    __m512 inv_r = _mm512_rsqrt14_ps(r2_vec);
        

    __m512 inv_r3 = _mm512_mul_ps(_mm512_mul_ps(inv_r, inv_r), inv_r);


    __m512 dfx_vec = _mm512_mul_ps(_mm512_mul_ps(rx_vec, m_vec), inv_r3);
    __m512 dfy_vec = _mm512_mul_ps(_mm512_mul_ps(ry_vec, m_vec), inv_r3);

    fx[i] -= _mm512_mask_reduce_add_ps(mask, dfx_vec); 
    fy[i] -= _mm512_mask_reduce_add_ps(mask, dfy_vec);

    printf("%d %g %g\n", i, fx[i], fy[i]);
  }

}
