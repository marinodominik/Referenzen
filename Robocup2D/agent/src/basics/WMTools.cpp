#include "basics/WMTools.h"

/* First half of the ASCII-table

  | 00  NUL '\0' | 10  DLE | 20  SPACE | 30  0 | 40  @ | 50  P        | 60  ` | 70  p   |
  | 01  SOH      | 11  DC1 | 21  !     | 31  1 | 41  A | 51  Q        | 61  a | 71  q   |
  | 02  STX      | 12  DC2 | 22  "     | 32  2 | 42  B | 52  R        | 62  b | 72  r   |
  | 03  ETX      | 13  DC3 | 23  #     | 33  3 | 43  C | 53  S        | 63  c | 73  s   |
  | 04  EOT      | 14  DC4 | 24  $     | 34  4 | 44  D | 54  T        | 64  d | 74  t   |
  | 05  ENQ      | 15  NAK | 25  %     | 35  5 | 45  E | 55  U        | 65  e | 75  u   |
  | 06  ACK      | 16  SYN | 26  &     | 36  6 | 46  F | 56  V        | 66  f | 76  v   |
  | 07  BEL '\a' | 17  ETB | 27  '     | 37  7 | 47  G | 57  W        | 67  g | 77  w   |
  | 08  BS  '\b' | 18  CAN | 28  (     | 38  8 | 48  H | 58  X        | 68  h | 78  x   |
  | 09  HT  '\t' | 19  EM  | 29  )     | 39  9 | 49  I | 59  Y        | 69  i | 79  y   |
  | 0A  LF  '\n' | 1A  SUB | 2A  *     | 3A  : | 4A  J | 5A  Z        | 6A  j | 7A  z   |
  | 0B  VT  '\v' | 1B  ESC | 2B  +     | 3B  ; | 4B  K | 5B  [        | 6B  k | 7B  {   |
  | 0C  FF  '\f' | 1C  FS  | 2C  ,     | 3C  < | 4C  L | 5C  \   '\\' | 6C  l | 7C  |   |
  | 0D  CR  '\r' | 1D  GS  | 2D  -     | 3D  = | 4D  M | 5D  ]        | 6D  m | 7D  }   |
  | 0E  SO       | 1E  RS  | 2E  .     | 3E  > | 4E  N | 5E  ^        | 6E  n | 7E  ~   |
  | 0F  SI       | 1F  US  | 2F  /     | 3F  ? | 4F  O | 5F  _        | 6F  o | 7F  DEL |

*/



/*       0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15*/
const unsigned char WMTools::enc_tab[ 64 ] =
{
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
        'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
        'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-', '+'
};

const int WMTools::XXXX = 0xFF;

const int WMTools::dec_tab[ 128 ] =
{
        XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,
        XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,
        XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,0x3F,XXXX,0x3E,XXXX,XXXX,
        0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,XXXX,XXXX,XXXX,XXXX,XXXX,XXXX,
        XXXX,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F,0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,
        0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F,0x20,0x21,0x22,0x23,XXXX,XXXX,XXXX,XXXX,XXXX,
        XXXX,0x24,0x25,0x26,0x27,0x28,0x29,0x2A,0x2B,0x2C,0x2D,0x2E,0x2F,0x30,0x31,0x32,
        0x33,0x34,0x35,0x36,0x37,0x38,0x39,0x3A,0x3B,0x3C,0x3D,XXXX,XXXX,XXXX,XXXX,XXXX
};

ANGLE WMTools::conv_server_angle_to_angle_0_2Pi( double s_ang )
{
    return ANGLE( ( -s_ang * M_PI ) / 180.0 );
}

/** singned 18 bit integer to 3 byte ascii using 64 fix characters

 this function encodes an integer in the range [2^18-1, ..., 2^18-1]
 other integers are treated as 2^18, which represents the infinity value

 the pointer dst should point to a memeory with 3 bytes free after it! */
bool WMTools::int18_to_a3x64( int src, char *dst )
{
    static const int mRANGE = -131071;
    static const int pRANGE =  131071;
    static const int infty  =  131072;

    bool res = true;

    if( src < mRANGE || src > pRANGE )
    {
        res = false;
        src = infty;
    }

    src -= mRANGE;

    int a = src % 64;
    int c = src / 64;
    int b = c   % 64;

    c /= 64;

    dst[ 0 ] = enc_tab[ c ];
    dst[ 1 ] = enc_tab[ b ];
    dst[ 2 ] = enc_tab[ a ];

    return res;
}

/** 3 byte ascii using 64 fix characters to signed integer with max 18 bit
 The value 2^18 indicates an overflow!

 the pointer src should point to a memeory with 3 bytes free after it and
 with valid characters. */
bool WMTools::a3x64_to_int18( const char *src, int &a )
{
    static const int mRANGE = -131071;
    //static const int pRANGE =  131071;
    //static const int infty  =  131072;

    int tmp;

    a = dec_tab[ ( unsigned char ) src[ 0 ] ];

    if( a == XXXX )
        return false;

    a *= 64;
    tmp = dec_tab[ ( unsigned char ) src[ 1 ] ];

    if( tmp == XXXX )
        return false;

    a += tmp;
    a *= 64;
    tmp = dec_tab[ ( unsigned char ) src[ 2 ] ];

    if( tmp == XXXX )
        return false;

    a += tmp;
    a += mRANGE;

    return true;
}

bool WMTools::uint6_to_a64( int src, char *dst )
{
    bool res = true;

    if( src < 0 || src > 63 )
    {
        res = false;
        src = 63;
    }

    dst[ 0 ] = enc_tab[ src ];

    return res;
}

bool WMTools::a64_to_uint6( const char *src, int &dst )
{
    dst = dec_tab[ ( unsigned char ) src[ 0 ] ];

    if( dst == XXXX )
        return false;

    return true;
}
