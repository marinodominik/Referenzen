#include "CommandsInfo.h"

#include "messages_times.h"

CommandsInfo::CommandsInfo()
{
    last_sb_time = -1;
}

CommandsInfo::~CommandsInfo()
{

}

void CommandsInfo::reset()
{
    last_sb_time = 0;
    for( int i = 0; i < CMD_MAX; i++ )
    {
        cmd_counter[ i ] = 0;
        cmd_send_time[ i ] = -10;
    }
}

void CommandsInfo::use_msg_sense_body( const Msg_sense_body &sb )
{
    int dum[ CMD_MAX ];

    dum[ CMD_MAIN_MOVETO ] = sb.move_count;
    dum[ CMD_MAIN_TURN ] = sb.turn_count;
    dum[ CMD_MAIN_DASH ] = sb.dash_count;
    dum[ CMD_MAIN_KICK ] = sb.kick_count;
    dum[ CMD_MAIN_CATCH ] = sb.catch_count;
    dum[ CMD_NECK_TURN ] = sb.turn_neck_count;
    dum[ CMD_SAY ] = sb.say_count;
    dum[ CMD_VIEW_CHANGE ] = sb.change_view_count;

    for( int i = 0; i < CMD_MAX; i++ )
    {
        if( last_sb_time + 1 == sb.time && cmd_send_time[ i ] == last_sb_time && dum[ i ] != cmd_counter[ i ] + 1 )
        {
            MessagesTimes::add_lost_cmd( sb.time, i );
        }

        cmd_counter[ i ] = dum[ i ];
    }

    last_sb_time = sb.time;
}

void CommandsInfo::set_command( int time, const Cmd &cmd )
{
    if( cmd.cmd_body.is_cmd_set() )
    {
        int cmd_type = CMD_INVALID;

        switch( cmd.cmd_body.get_type() )
        {
            case Cmd_Body::TYPE_MOVETO :
                cmd_type = CMD_MAIN_MOVETO;
                break;
            case Cmd_Body::TYPE_TURN :
                cmd_type = CMD_MAIN_TURN;
                break;
            case Cmd_Body::TYPE_DASH :
                cmd_type = CMD_MAIN_DASH;
                break;
            case Cmd_Body::TYPE_KICK :
                cmd_type = CMD_MAIN_KICK;
                break;
            case Cmd_Body::TYPE_CATCH :
                cmd_type = CMD_MAIN_CATCH;
                break;
            case Cmd_Body::TYPE_TACKLE :
                cmd_type = CMD_MAIN_TACKLE;
                break;
            default :
                ERROR_OUT << ID << "\nwrong type";
        }

        cmd_send_time[ cmd_type ] = time;
        MessagesTimes::add_sent_cmd( time, cmd_type );
    }

    if( cmd.cmd_neck.is_cmd_set() )
        cmd_send_time[ CMD_NECK_TURN ] = time;

    if( cmd.cmd_say.is_cmd_set() )
        cmd_send_time[ CMD_SAY ] = time;

    if( cmd.cmd_view.is_cmd_set() )
    {
        cmd_send_time[ CMD_VIEW_CHANGE ] = time;
        MessagesTimes::add_sent_cmd( time, CMD_VIEW_CHANGE );
    }
}
