#include "xls_int.h"
#pragma hls_top
XlsInt<9, false> add8(XlsInt<8, false> a, XlsInt<8, false> b) {
  return XlsInt<9, false>(a) + XlsInt<9, false>(b);
}
