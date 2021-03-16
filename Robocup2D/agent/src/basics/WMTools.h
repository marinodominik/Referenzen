#ifndef _WMTOOLS_H_
#define _WMTOOLS_H_

#include <math.h>
#include "angle.h"

class WMTools
{

private:

    static const unsigned char enc_tab[ 64 ];
    static const int XXXX;
    static const int dec_tab[ 128 ];

public:

    static ANGLE conv_server_angle_to_angle_0_2Pi( double s_ang );

    /** singned 18 bit integer to 3 byte ascii using 64 fix characters

     this function encodes an integer in the range [2^18-1, ..., 2^18-1]
     other integers are treated as 2^18, which represents the infinity value

     the pointer dst should point to a memeory with 3 bytes free after it! */
    static bool int18_to_a3x64( int src, char *dst );

    /** 3 byte ascii using 64 fix characters to signed integer with max 18 bit
     The value 2^18 indicates an overflow!

     the pointer src should point to a memeory with 3 bytes free after it and
     with valid characters. */
    static bool a3x64_to_int18( const char *src, int &a );

    /// [0,...,62] -> ascii char
    static bool uint6_to_a64( int src, char *dst );

    /// ascii char to [0,...,62], 63 indicates overflow
    static bool a64_to_uint6( const char *src, int &dst );
};

#endif
