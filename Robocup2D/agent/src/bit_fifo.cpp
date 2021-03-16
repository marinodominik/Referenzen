#include "bit_fifo.h"

BitFIFO::BitFIFO()
{
    reset();
}

void BitFIFO::reset()
{
    first = 0;
    next_free = 0;
}

int BitFIFO::get_free_size()
{
    return MAX_NUM_BITS - get_size();
}

int BitFIFO::get_size()
{
    if( next_free >= first )
        return next_free-first;

    return MAX_NUM_BITS - first + next_free;
}

bool BitFIFO::put( int num_bits, unsigned int in )
{
    for( int i = 0; i < num_bits; i++ )
    {
        tab[ next_free++ ] = in & 1;
        in >>= 1;

        if( next_free >= MAX_NUM_BITS )
            next_free= 0;

        if( next_free == first )
            return false;
    }

    return true;
}

bool BitFIFO::fill_with_zeros( int num_bits )
{
    for( int i = 0; i < num_bits; i++ )
    {
        tab[ next_free++ ] = 0;
        if( next_free >= MAX_NUM_BITS )
            next_free = 0;

        if( next_free == first )
            return false;
    }

    return true;
}

bool BitFIFO::get( int num_bits, unsigned int &output )
{
    int idx = first + num_bits;

    if( idx >= MAX_NUM_BITS )
        idx -= MAX_NUM_BITS;

    int new_first = idx;

    unsigned int out = 0;

    for( int i = 0; i < num_bits; i++ )
    {
        if( idx == 0 )
            idx = MAX_NUM_BITS;

        idx--;

    if( idx == next_free )
        return false;

    out <<= 1;

    if( tab[ idx ] )
        out |= 1;
    }

    first = new_first;

    output = out;

    return true;
}

void BitFIFO::show( std::ostream &out ) const
{
    out << "[" << first << ":" << next_free <<  " |";

    int idx = first;

    while( idx != next_free )
    {
        if( idx % 5 == 0 )
            out << ' ';

        out << tab[ idx++ ];

        if( idx == MAX_NUM_BITS )
            idx = 0;
    }

    out << "]";
}
