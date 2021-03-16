#ifndef BS2K_BASICS_BALL_H_
#define BS2K_BASICS_BALL_H_

#include <limits>

#include "Vector.h"

struct Ball
{
    int time;
    int age;
    int age_vel;

    Vector pos;
    Vector vel;
    int invalidated; // 1 means ball was invalidated in current cycle, 2 means (possibly much) more the one cycle!

    Ball();
};

std::ostream& operator<<( std::ostream &outStream, const Ball &ball );

#endif /* BS2K_BASICS_BALL_H_ */
