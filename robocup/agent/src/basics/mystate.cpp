#include "mystate.h"

void MyState::get_from_WS()
{
    my_vel   = WSinfo::me->vel;
    my_pos   = WSinfo::me->pos;
    ball_vel = WSinfo::ball->vel;
    ball_pos = WSinfo::ball->pos;
    my_angle = WSinfo::me->ang;
}

MyState::MyState()
{
    op = NULL;
    me = WSinfo::me;
}
