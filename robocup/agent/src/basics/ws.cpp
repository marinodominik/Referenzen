#include "ws.h"

WS::WS()
{
    time                 = -1;
    time_of_last_update  = -1;
    ms_time_of_see_delay = -1;
    ms_time_of_sb        = -1;
    ms_time_of_see       = -1;

    play_mode      = PM_Unknown;
    penalty_count  = -1;
    my_team_score  = -1;
    his_team_score = -1;

    my_goalie_number  = -1;
    his_goalie_number = -1;

    my_attentionto = -1;

    current_direct_opponent_assignment           = -1;
    number_of_direct_opponent_updates_from_coach = -1;

    synch_mode_ok = false;
    view_angle    = -1;
    view_quality  = -1;

    last_message.heart_at = -1;
    last_message.sent_at  = -1;
    last_message.from     =  0;

    last_successful_say_at = -1;

    my_team_num  = -1;
    his_team_num = -1;
}

std::ostream& operator<<( std::ostream &outStream, const WS &ws )
{
    outStream << "\nWSstate     --------------------------------"
              << "\ntime           = " << ws.time
              << "\nplay_mode      = " << play_mode_str(ws.play_mode)
              << "\n--"
              << "\nmy_team_score  = " << ws.my_team_score
              << "\nhis_team_score = " << ws.his_team_score
              << "\n--"
              << "\nsynch view     = " << ws.synch_mode_ok
              << "\nview_angle     = ";

    switch( ws.view_angle )
    {
    case WIDE   : outStream << "WIDE";          break;
    case NARROW : outStream << "NARROW";        break;
    case NORMAL : outStream << "NORMAL";        break;
    default     : outStream << "invalid value";
    }

    outStream << "\nview_quality   = ";

    switch( ws.view_quality )
    {
    case HIGH : outStream << "HIGH";          break;
    case LOW  : outStream << "LOW";           break;
    default   : outStream << "invalid value";
    }

    outStream << "\n--"
              << "\nball           = " << ws.ball

              << "\n--"
              << "\nmy_team (" << ws.my_team_num << ")        = ";
    for( int i = 0; i < ws.my_team_num; i++)
    {
        outStream << ws.my_team[ i ];
    }

    outStream << "\n--"
              << "\nhis_team (" << ws.his_team_num << ")      = ";
    for( int i = 0; i < ws.his_team_num; i++)
    {
        outStream << ws.his_team[ i ];
    }

    outStream << "\nWSstate end --------------------------------";

    return outStream;
}
