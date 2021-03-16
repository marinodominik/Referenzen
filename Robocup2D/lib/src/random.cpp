#include "random.h"
#include <climits>

using namespace Statistics;

unsigned long int Statistics::Random::seed;

void Random::setSeed( unsigned long int newSeed ) throw ()
{
    seed = newSeed;
}

unsigned long int Random::getSeed() throw ()
{
    return seed;
}

double Random::basic_random() throw ()
{
    // we assume that an unsigned long int has 32 bits

    seed = seed * 69069 + 1;

    return ( static_cast < double > ( seed ) / ULONG_MAX );
}

