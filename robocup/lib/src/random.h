// File random.h:
// contains the declaration of random number generators
// created 06-OCT-00 by Martin Lauer
// ---------------------------------------------

#ifndef random_h
#define random_h

namespace Statistics
{
    class Random
    {
    private:
        static unsigned long int seed;
    public:
        static void setSeed( unsigned long int newSeed ) throw();     // set seed
        static unsigned long int getSeed() throw();         // get seed
        static double basic_random() throw();            // random number in [0,1]
    };
}

#endif
