#include "Ball.h"

Ball::Ball()
{
    time        = -1;
    age         = std::numeric_limits< int >::max();
    age_vel     = std::numeric_limits< int >::max();
    invalidated = 0;
}

std::ostream& operator<<( std::ostream &outStream, const Ball &ball )
{
    return outStream
            << "\nBall "
            << " age= "  << ball.age
            << " time= " << ball.time
            << ", Pos= " << ball.pos
            << ", Vel= " << ball.vel;
}
