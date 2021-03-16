#include "MessagesInfo.h"

MessagesInfo::MessagesInfo()
{

}

MessagesInfo::~MessagesInfo()
{

}

void MessagesInfo::reset()
{
    for( int i = 0; i < MESSAGE_MAX; i++ )
    {
        msg[ i ].received  = 0;
        msg[ i ].processed = false;
    }
}

void MessagesInfo::set_cycle( int c )
{
    for( int i = 0; i < MESSAGE_MAX; i++ )
        msg[ i ].cycle = c;
}

void MessagesInfo::set_ms_time( long ms_time )
{
    for( int i = 0; i < MESSAGE_MAX; i++ )
        msg[ i ].ms_time = ms_time;
}
