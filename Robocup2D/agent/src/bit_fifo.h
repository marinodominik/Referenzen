#ifndef _BIT_FIFO_H_
#define _BIT_FIFO_H_

#include <iostream>

class BitFIFO
{
    static const int MAX_NUM_BITS = 100;

    int first;
    int next_free;

    bool tab[ MAX_NUM_BITS ];

    int get_free_size();
public:
    BitFIFO();

    int get_size();
    void reset();

    bool put( int num_bits, unsigned int in );
    bool get( int num_bits, unsigned int &out );

    void show( std::ostream &out ) const;

    bool fill_with_zeros( int num_bits );
};

#endif
