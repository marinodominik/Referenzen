#include "sensorbuffer.h"

std::ostream& operator<<( std::ostream &outStream, const Msg_sense_body &sb )
{
    outStream << "\nMsg_sense_body "
            << "\n  time         " << sb.time;

    outStream << "\n  view_quality ";
    switch( sb.view_quality )
    {
        case HIGH: outStream << "HIGH";          break;
        case LOW:  outStream << "LOW";           break;
        default:   outStream << "invalid value";
    }

    outStream << "\n  view_width   ";
    switch( sb.view_width )
    {
        case WIDE:   outStream << "WIDE";          break;
        case NARROW: outStream << "NARROW";        break;
        case NORMAL: outStream << "NORMAL";        break;
        default:     outStream << "invalid value";
    }

    outStream << "\n  stamina      " << sb.stamina
            << "\n  stamina cap  " << sb.stamina_capacity
            << "\n  effort       " << sb.effort
            << "\n  speed_value  " << sb.speed_value
            << "\n  speed_angle  " << sb.speed_angle
            << "\n  neck_angle   " << sb.neck_angle
            << "\n  #k=          " << sb.kick_count
            << " #d=             " << sb.dash_count
            << " #t=             " << sb.turn_count
            << " #s=             " << sb.say_count
            << " #t_neck=        " << sb.turn_neck_count
            << " #c=             " << sb.catch_count
            << " #m=             " << sb.move_count
            << " #c_view=        " << sb.change_view_count;

    return outStream;
}

std::ostream& operator<<( std::ostream &outStream, const Msg_see &see )
{
    outStream << "\nMsg_see "
            << "\n  time= " << see.time;

    if( see.markers_num > 0 )
    {
        outStream << "\n  Markers (" << see.markers_num << ")";
        for( int i = 0; i < see.markers_num; i++ )
        {
            outStream << "\n  ( "       << see.markers[ i ].x << "," << see.markers[ i ].y << ")"
                    << " #= "           << see.markers[ i ].how_many
                    << " dist= "        << see.markers[ i ].dist
                    << " dir= "         << see.markers[ i ].dir
                    << " dist_change= " << see.markers[ i ].dist_change
                    << " dir_change= "  << see.markers[ i ].dir_change
                    << ")";
        }
    }

    if( see.players_num > 0 )
    {
        outStream << "\n  Players (" << see.players_num << ")";
        for( int i = 0; i < see.players_num; i++ )
        {
            outStream << "\n team= ";
            switch( see.players[ i ].team )
            {
            case my_TEAM:  outStream << "MY  "; break;
            case his_TEAM: outStream << "HIS "; break;
            default:       outStream << "< ? >";
            }
            outStream << " number= "    << see.players[ i ].number
                    << " goalie= "      << see.players[ i ].goalie
                    << " #= "           << see.players[ i ].how_many
                    << " dist= "        << see.players[ i ].dist
                    << " dir= "         << see.players[ i ].dir
                    << " dist_change= " << see.players[ i ].dist_change
                    << " dir_change= "  << see.players[ i ].dir_change
                    << " body_dir= "    << see.players[ i ].body_dir
                    << " head_dir= "    << see.players[ i ].head_dir;
        }
    }

    if( see.line_upd )
    {
        outStream << "\n  Line "
                << "\n  ( "         << see.line.x << "," << see.line.y << ")"
                << " #= "           << see.line.how_many
                << " dist= "        << see.line.dist
                << " dir= "         << see.line.dir
                << " dist_change= " << see.line.dist_change
                << " dir_change= "  << see.line.dir_change;
    }

    if( see.ball_upd )
    {
        outStream << "\n  Ball "
                << "\n "
                << " #= "           << see.ball.how_many
                << " dist= "        << see.ball.dist
                << " dir= "         << see.ball.dir
                << " dist_change= " << see.ball.dist_change
                << " dir_change= "  << see.ball.dir_change;
    }

    return outStream;
}

std::ostream& operator<<( std::ostream &outStream, const Msg_hear &hear )
{
    outStream << "\nMsg_hear "
            << "\n  time= " << hear.time;

    if( hear.play_mode_upd ) outStream << "\n  play_mode= " << show_play_mode( hear.play_mode );
    if( hear.my_score_upd )  outStream << "\n  my_score= " << hear.my_score;
    if( hear.his_score_upd ) outStream << "\n  his_score= " << hear.his_score;
    if( hear.teamcomm_upd )  outStream << "\n  heard communication: " << hear.teamcomm;

    return outStream;
}

std::ostream& operator<<( std::ostream &outStream, const Msg_init &init )
{
    outStream << "\nMsg_init "
            << "\n  side  = ";

    switch( init.side )
    {
    case left_SIDE:  outStream << "left"; break;
    case right_SIDE: outStream << "right"; break;
    default:         outStream << "< ? >";
    }

    outStream << "\n  number=    " << init.number
            << "\n  play_mode= "   << show_play_mode( init.play_mode );

    return outStream;
}

std::ostream& operator<<( std::ostream &outStream, const Msg_fullstate &fs )
{
    outStream << "\n Msg_fullstate "
            << "\n  time         = " << fs.time
            << "\n  play_mode    = " << show_play_mode( fs.play_mode );

    outStream << "\n  view_quality ";
    switch( fs.view_quality )
    {
    case HIGH: outStream << "HIGH";          break;
    case LOW: outStream  << "LOW";           break;
    default: outStream   << "invalid value";
    }

    outStream << "\n  view_width   ";
    switch( fs.view_width )
    {
    case WIDE:   outStream << "WIDE";          break;
    case NARROW: outStream << "NARROW";        break;
    case NORMAL: outStream << "NORMAL";        break;
    default:     outStream << "invalid value";
    }

    outStream << "\n  my_score     = " << fs.my_score
            << "\n  his_score    = "   << fs.his_score;

    if( fs.players_num > 0 )
    {
        outStream << "\n  Players (" << fs.players_num << ")";
        for( int i = 0; i < fs.players_num; i++ )
        {
            outStream << "\n  team= ";
            switch( fs.players[ i ].team )
            {
            case my_TEAM:  outStream << "MY  ";  break;
            case his_TEAM: outStream << "HIS ";  break;
            default:       outStream << "< ? >";
            }
            outStream << " number= "   << fs.players[ i ].number
                    << " pos= ("       << fs.players[ i ].x     << "," << fs.players[ i ].y     << ")"
                    << " vel= ("       << fs.players[ i ].vel_x << "," << fs.players[ i ].vel_y << ")"
                    << " angle= "      << fs.players[ i ].angle
                    << " neck_angle= " << fs.players[ i ].neck_angle
                    << " stamina "     << fs.players[ i ].stamina
                    << " effort "      << fs.players[ i ].effort
                    << " recovery "    << fs.players[ i ].recovery;
        }
    }

    outStream << "\n  Ball "
            << " pos= (" << fs.ball.x     << "," << fs.ball.y     << ")"
            << " vel= (" << fs.ball.vel_x << "," << fs.ball.vel_y << ")";

    return outStream;
}

std::ostream& operator<<( std::ostream &outStream, const Msg_fullstate_v8 &fs )
{
    outStream << "\n Msg_fullstate "
            << "\n  time         = " << fs.time
            << "\n  play_mode    = " << show_play_mode( fs.play_mode );

    outStream << "\n  view_quality ";
    switch( fs.view_quality )
    {
    case HIGH: outStream << "HIGH";          break;
    case LOW:  outStream << "LOW";           break;
    default:   outStream << "invalid value";
    }

    outStream << "\n  view_width   ";
    switch( fs.view_width )
    {
    case WIDE:   outStream<< "WIDE";           break;
    case NARROW: outStream<< "NARROW";         break;
    case NORMAL: outStream<< "NORMAL";         break;
    default:     outStream << "invalid value";
    }

    outStream << "\n  count        = "
            << " k:"  << fs.count_kick
            << " d:"  << fs.count_dash
            << " t:"  << fs.count_turn
            << " c:"  << fs.count_catch
            << " m:"  << fs.count_move
            << " tn:" << fs.count_turn_neck
            << " cv:" << fs.count_change_view
            << " s:"  << fs.count_say;

    outStream << "\n  my_score     = " << fs.my_score
            << "\n  his_score    = "   << fs.his_score;

    outStream << "\n  Ball "
            << " pos= (" << fs.ball.x     << "," << fs.ball.y     << ")"
            << " vel= (" << fs.ball.vel_x << "," << fs.ball.vel_y << ")";

    if( fs.players_num > 0 )
    {
        outStream << "\n  Players (" << fs.players_num << ")";
        for( int i = 0; i < fs.players_num; i++ )
        {
            outStream << "\n  team= ";
            switch( fs.players[ i ].team )
            {
            case my_TEAM:  outStream << "MY  ";  break;
            case his_TEAM: outStream << "HIS ";  break;
            default:       outStream << "< ? >";
            }

            if( fs.players[ i ].goalie ) outStream << " (goalie)";

            outStream << " number= "        << fs.players[ i ].number
                    << " pos= ("            << fs.players[ i ].x     << "," << fs.players[ i ].y     << ")"
                    << " vel= ("            << fs.players[ i ].vel_x << "," << fs.players[ i ].vel_y << ")"
                    << " angle= "           << fs.players[ i ].angle
                    << " neck_angle= "      << fs.players[ i ].neck_angle
                    << " stamina "          << fs.players[ i ].stamina
                    << " effort "           << fs.players[ i ].effort
                    << " recovery "         << fs.players[ i ].recovery
                    << " stamina_capacity " << fs.players[ i ].stamina_capacity;
        }
    }

    return outStream;
}

std::ostream& operator<<( std::ostream &outStream, const Msg_teamcomm &tc )
{
    outStream << "\n Msg_teamcomm "
            << "\n time       = " << tc.time
            << "\n time_cycle = " << tc.time_cycle
            << "\n from       = " << tc.from;

    if( tc.his_goalie_number_upd )
        outStream << "\n his_goalie_number= " << tc.his_goalie_number;
    else
        outStream << "\n no his_goalie_number";

    if( tc.ball_upd )
    {
        outStream << "\n ball"
                << "\n how_old= " << tc.ball.how_old
                <<" x= "          << tc.ball.x     << " y= "     << tc.ball.y
                << " vel_x= "     << tc.ball.vel_x << " vel_y= " << tc.ball.vel_y;
    }
    else
    {
        outStream << "\n no ball";
    }

    if( tc.players_num > 0 )
    {
        outStream << "\n Players (" << tc.players_num << ")";
        for( int i = 0; i < tc.players_num; i++ )
        {
            outStream << "\n how_old= " << tc.players[ i ].how_old
                    << " team= ";
            switch( tc.players[ i ].team )
            {
            case my_TEAM:  outStream << "MY  ";  break;
            case his_TEAM: outStream << "HIS ";  break;
            default:       outStream << "< ? >";
            }
            outStream << " number= " << tc.players[ i ].number
                    << " pos= ("     << tc.players[ i ].x      << "," << tc.players[ i ].y << ")";
        }
    }

    return outStream;
}

std::ostream& operator<<( std::ostream &outStream, const Msg_teamcomm2 &tc )
{
    outStream << " Msg_teamcomm2 ";

    outStream << "\n";
    if( tc.msg.valid )
    {
        outStream << " | msg"
                << " from= "  << tc.msg.from
                << " p1= "    << tc.msg.param1
                << " p2= "    << tc.msg.param2;
    }
    else
    {
        outStream << " | no msg";
    }

    outStream << "\n";
    if( tc.ball.valid )
    {
        outStream << " | ball " << tc.ball.pos;
    }
    else
    {
        outStream << " | no ball";
    }

    outStream << "\n";
    if( tc.pass_info.valid )
    {
        outStream << " | pass "
                << "  pos= " << tc.pass_info.ball_pos
                << "  vel= " << tc.pass_info.ball_vel
                << " t= "    << tc.pass_info.time;
    }
    else
    {
        outStream << " | no pass info";
    }

    outStream << "\n";
    if( tc.ball_info.valid )
    {
        outStream << " | ball_info "
                << "  pos= " << tc.ball_info.ball_pos
                << "  vel= " << tc.ball_info.ball_vel
                << "  ap= "  << tc.ball_info.age_pos
                << "  av= "  << tc.ball_info.age_vel;
    }
    else
    {
        outStream << " | no ball info";
    }

    if( tc.ball_holder_info.valid )
    {
        outStream << " | ball_holder "
                << "  pos= " << tc.ball_holder_info.pos;
    }
    else
    {
        outStream << " |no ball_holder";
    }

    outStream << "\n";
    if( tc.players_num > 0 )
    {
        outStream << " | Players (" << tc.players_num << ")";
        for( int i = 0; i < tc.players_num; i++ )
        {
            outStream << " | team= ";
            switch( tc.players[ i ].team )
            {
            case my_TEAM:  outStream <<  "MY  "; break;
            case his_TEAM: outStream << "HIS ";  break;
            default:       outStream << "< ? >";
            }
            outStream << " number= " << tc.players[ i ].number
                    << " pos= "      << tc.players[ i ].pos;
        }
    }
    else
    {
        outStream << " | no players";
    }

  return outStream;
}

std::ostream& operator<<( std::ostream &outStream, const Msg_my_online_coachcomm &moc )
{
    outStream << "\n Msg_my_online_coachcomm "
            << "\n time= " << moc.time;

    if( !moc.his_player_types_upd )
    {
        outStream << "\n his_player_types_upd= false";
    }
    else
    {
        outStream << "\n his_player_types=";
        for( int i = 0; i < NUM_PLAYERS; i++ )
        {
            outStream << " p_" << i + 1 << ":" << moc.his_player_types[ i ];
        }
    }

    if( !moc.direct_opponent_assignment_upd )
    {
    outStream << "\n direct_opponent_assignment_upd = false";
    }
    else
    {
        outStream << "\n direct_opponent_assignment =";
        for( int i = 0; i < NUM_PLAYERS; i++ )
        {
            outStream << " p_" << i + 1 << ":" << moc.direct_opponent_assignment[ i ];
        }
    }

    return outStream;
}
