#include "Communicator.h"

#include <sstream>
#include <sys/select.h>
#include <errno.h> //for return values of select

#include "str2val.h"

#include "log_macros.h"
#include "tools.h"

#include "serverparam.h"
#include "default_server_param.h"
#include "sensorparser.h"
#include "ws_memory.h"

#include "messages_times.h"
#include "blackboard.h"

#include <string>
#include <stdlib.h>

Communicator::Communicator()
{
    got_sock_data = false;
}

Communicator::~Communicator()
{

}

bool Communicator::initialize()
{
    return sock.init_socket_fd() && sock.init_serv_addr( CommandLineOptions::host, CommandLineOptions::port );
}

#ifdef CMD_VIEW_HACK
void Communicator::set_cmd_view_hack(Cmd_View cmd_view)
{
    this->cmd_view_hack = cmd_view;
}
#endif

bool Communicator::has_data_on_socket()
{
    return got_sock_data;
}

void Communicator::set_socket_nonblocking()
{
    sock.set_fd_nonblock();
}

/** waits until the given time is over, or until something happens with the file descriptor

\param
 */
int Communicator::idle(int ms_time, int s_time)
{
    fd_set rfds;
    struct timeval tv;
    int retval;
    int max_fd_plus_1= sock.get_socket_fd() + 1;
    got_sock_data= false;

    FD_ZERO(&rfds);

    //FD_SET(0, &rfds); //Standardeingabe
    FD_SET(sock.get_socket_fd(), &rfds);

    tv.tv_usec= 1000 * ms_time;
    tv.tv_sec = s_time;

    //long ms_time_before= Tools::get_current_ms_time();//TEST, muss weg!!!!

    retval = select(max_fd_plus_1, &rfds, NULL, NULL, &tv);
    /* Don't rely on the value of tv now! */

#if 0 //test if select waits longer the required
    long ms_time_after= Tools::get_current_ms_time(); //TEST, muss weg!!!!
    long tmp= 1000 - (tv.tv_usec / 1000);
    //cerr << "\n+++" << tv.tv_usec << ", " << tmp << " ms_time= " <<ms_time;
    if ( s_time <= 0 && ms_time_after - ms_time_before > ms_time + 10)
    {
        //cerr << "\n*** tv.tv_usec / 1000= " << tmp << " ms_time= " << ms_time << " diff= " << tmp-ms_time << flush;
        cerr << "\n*** tv.tv_usec / 1000= " << tmp << " ms_time= " << ms_time << " diff= " << ms_time_after - ms_time_before - ms_time;
    }
#endif
    if ( retval > 0 ){
        if ( FD_ISSET(sock.get_socket_fd(),&rfds) )
        {
            got_sock_data= true;
            return 1;
        }
        else
            ERROR_OUT << ID << "\nwarning: received something, but not on the fd= " << sock.get_socket_fd() ;
    }

    if (retval < 0)
    {
        switch (errno)
        {
        case EBADF:
            ERROR_OUT << ID << "\nselect error: an invalid file descriptor was given in one of the sets";
            break;
        case EINTR:
            //cerr << "\nselect error: a non blocked signal was caught";
            break;
        case EINVAL:
            ERROR_OUT << ID << "\nselect error: n is negative";
            break;
        case ENOMEM:
            ERROR_OUT << ID << "\nselect error: select was unable to allocate memory for internal tables";
            break;
        default:
            ERROR_OUT << ID << "\nselect error: error code " << errno;
        }
        return -1;
    }

    return 0;
}

/** returns the number of read messages */
int Communicator::receive_and_incorporate_server_messages( MessagesInfo & msg_info, CommandsInfo & cmd_info, WMstate * state, WMstate * state_full, BodyBehavior *body_controller, Cmd& last_cmd )
{
    int num_messages = 0;
    int num_bytes;
    Msg_sense_body         sense_body;
    Msg_see                see;
    Msg_fullstate          fullstate;
    Msg_fullstate_v8       fullstate_v8;
    Msg_hear               hear;
    Msg_change_player_type change_player_type;
    //Msg_teamcomm2 tc;
    //Msg_server_param server_param;

    int type, time;

    bool reset_int = false;

    long ms_time = Tools::get_current_ms_time();



    while( sock.recv_msg( buffer,num_bytes ) ) //read all messages from the socket and incorporate them!
    {
        buffer[ num_bytes ] = '\000';
        num_messages++;

        ms_time = Tools::get_current_ms_time(); //seems to be better if time is measured every time

        bool res = SensorParser::get_message_type_and_time(buffer,type,time);

        if( type == MESSAGE_THINK && time < 0 ) //MESSAGE_THINK doesn't have own time stamp
            time= msg_info.msg[MESSAGE_SENSE_BODY].cycle;

        if (!res)
        {
            ERROR_OUT << ID << "\nwrong message type " << buffer;
            return -1;
        }

        MessagesTimes::add( time, ms_time, type );

        msg_info.msg[ type ].received++;
        msg_info.msg[ type ].cycle   = time;
        msg_info.msg[ type ].ms_time = ms_time;

        const char *error_point= 0;



        switch( type )
        {

        case MESSAGE_SENSE_BODY:

            WM::time = time;

            if(WMoptions::ignore_sense_body)
                break;

            msg_info.msg[ type ].processed = true;

            res = SensorParser::manual_parse_sense_body( buffer, sense_body, error_point );
            if( !res )
                break;

            cmd_info.use_msg_sense_body( sense_body );

            if( state ) state->incorporate_cmd_and_msg_sense_body( last_cmd, sense_body );
            break;



        case MESSAGE_SEE:

            if( WMoptions::ignore_see )
                break;

            msg_info.msg[ type ].processed = true;

            res = SensorParser::manual_parse_see( buffer, see, error_point );
            if( !res )
                break;

            LOG_WM( see.time, 0, << "MSG SEE: " << msg_info.msg[ MESSAGE_SEE ].ms_time - msg_info.msg[ MESSAGE_SENSE_BODY ].ms_time << " ms after sense body  now=" << ms_time ); // << "\n" << buffer );

            if( state )
            {
                //bool tc_was_sent= state->time == state->time_of_last_msg_see;
                state->incorporate_msg_see( see,
                        msg_info.msg[ MESSAGE_SEE        ].ms_time,
                        msg_info.msg[ MESSAGE_SEE        ].ms_time - msg_info.msg[ MESSAGE_SENSE_BODY ].ms_time);

#ifdef CMD_VIEW_HACK
                if( !WMoptions::behave_after_think ) //do not use view mode synchronization in synch_mode
                    if( cmd_view_hack.is_cmd_set() )
                    {
                        send_cmd_view( cmd_view_hack );
                        cmd_view_hack.unset_lock();
                        cmd_view_hack.unset_cmd();
                        if( cmd_view_hack.is_cmd_set() )
                        {
                            ERROR_OUT << ID << "this should never be true";
                        }
                    }
#endif

            }
            break;



        case MESSAGE_FULLSTATE:

            if( WMoptions::ignore_fullstate )
                break;

            msg_info.msg[ type ].processed = true;

            if( ClientOptions::server_version >= 8.0 )
            {
                res = SensorParser::manual_parse_fullstate( buffer, fullstate_v8 );
                if( !res )
                    break;

                if( state_full )
                {
                    state_full->import_msg_fullstate( fullstate_v8 );
                }
                LOG_WM( fullstate_v8.time, 0, << " $$$$$ FULL my vel= " << state_full->my_team[ WM::my_number ].vel << " value= " << state_full->my_team[ WM::my_number ].vel.norm() << " ang= " << state_full->my_team[ WM::my_number ].vel.arg() );
            }
            else
            {
                res = SensorParser::manual_parse_fullstate( buffer, fullstate );
                if( !res )
                    break;

                if( state_full ) state_full->import_msg_fullstate( fullstate );
                LOG_WM( fullstate.time, 0, << " $$$$$ FULL my vel= " << state_full->my_team[ WM::my_number ].vel << " value= " << state_full->my_team[ WM::my_number ].vel.norm() << " ang= " << state_full->my_team[ WM::my_number ].vel.arg() );
            }

            break;



        case MESSAGE_THINK:
            msg_info.msg[ type ].processed = true;
            break;



        case MESSAGE_HEAR:

            if( WMoptions::ignore_hear )
                break;

            msg_info.msg[ type ].processed = true;

            reset_int = false;

            SensorParser::manual_parse_hear( buffer, hear, error_point, reset_int );

            if( reset_int )
            {
                body_controller->reset_intention();
                WSmemory::init();
                Blackboard::init();
            }

            if( !res )
                break;

            if( state      )      state->incorporate_msg_hear( hear );
            if( state_full ) state_full->incorporate_msg_hear( hear );
            break;



        case MESSAGE_CHANGE_PLAYER_TYPE:
            msg_info.msg[ type ].processed = true;
            res = SensorParser::manual_parse_change_player_type( buffer, change_player_type );

            if( !res )
                break;

            if( state ) state->incorporate_msg_change_player_type( change_player_type );
            break;



        case MESSAGE_INIT:
            ERROR_OUT << ID << "\nreceived init message: " << buffer << flush;
            break;



        case MESSAGE_SERVER_PARAM:
            //cerr << "\nreceived server_param message (not yet supported)" << flush;
            break;



        case MESSAGE_PLAYER_PARAM:
            //cerr << "\nreceived player_param message (not yet supported)" << flush;
            break;



        case MESSAGE_PLAYER_TYPE:
            //cerr << "\nreceived player_type message (not yet supported)" << flush;
            break;



        case MESSAGE_OK:
            if( strcmp( buffer, "(ok synch_see)" ) == 0 )
            {
                // if we get that message we act in synched view mode from now on
                state->synch_mode_ok = true;
            };
            INFO_OUT << ID << "\nreceived ok message: " << buffer << flush;
            break;



        case MESSAGE_ERROR:
            INFO_OUT << ID << "\nreceived error message [p=" << WM::my_number << " t= " << state->time.time << "]: "<< buffer << flush;
            break;



        default:
            ERROR_OUT << ID << "\nunknown message: " << buffer << flush;

        } // end of switch

        if( !res )
        {
            ERROR_OUT << ID << "\nproblems with message: ";
            SensorParser::show_parser_error_point( ERROR_STREAM, buffer, error_point );
        }
    }

    return num_messages;
}

void Communicator::send_initialize_message()
{
    if( WMoptions::offline )
        return;

    ostringstream initMessage;

    initMessage << "(init " << ClientOptions::teamname
                << " (version " << ClientOptions::server_version << ")";
    if( ClientOptions::consider_goalie )
        initMessage << " (goalie)";
    initMessage << ")" << ends;

    sock.send_msg( initMessage.str() );
}

bool Communicator::recv_initialize_message( PlayMode &pm )
{
    if( WMoptions::offline )
    {
        WM::my_side   = left_SIDE;
        WM::my_number = 2;
        return true;
    }

    int num_bytes;
    if( sock.recv_msg( buffer, num_bytes, true ) ) //true = does redirect to new port address (differs from 6000)
    {
        buffer[ num_bytes ] = '\0';
        //cout << "\nreceived: " << buffer << flush;
    }

    int type, time;
    bool res = SensorParser::get_message_type_and_time( buffer, type, time );

    if( type != MESSAGE_INIT )
        res = false;

    Msg_init init;
    if( res )
        res = SensorParser::manual_parse_init( buffer, init );

    if( !res )
    {
        ERROR_OUT << ID << "\nWRONG INITIALIZATION STRING " << buffer << endl;
        return false;
    }

    //wm.play_mode = init.play_mode; //important to get the initial play_mode
    pm = init.play_mode; //important to get the initial play_mode

    WM::my_side   = init.side;
    WM::my_number = init.number;

    sock.send_msg( "(ear (off opp))" ); //don't want to hear opponent's messages !!!

    //"(ear (off opp))(ear (off our partial))(ear (on our complete))"; //don't want to hear opponent's messages, don't want to hear partial messages !!!
    //"(ear (off opp))(ear (on our complete))"; //don't want to hear opponent's messages !!!

    return true;
}

bool Communicator::recv_parameter_messages()
{
    bool res;

    if( WMoptions::offline )
    {
        res =    ServerParam::incorporate_server_param_string( DEFAULT_MESSAGE_SERVER_PARAM )
        && ServerParam::incorporate_player_param_string( DEFAULT_MESSAGE_PLAYER_PARAM);

        for( int i = 0; i < DEFAULT_NUM_MESSAGE_PLAYER_TYPE; i++ )
            res &= ServerParam::incorporate_player_type_string( DEFAULT_MESSAGE_PLAYER_TYPE[ i ] );

        return res;
    }

    //the socket should be blocking in this routine
    // message_count_down prevents the function to soak up
    // infinitely many messages in case not all params were set
    // to suffice ServerParam::all_params_ok()
    // (theoretically we can ony get 20 messages here)
    res = false;
    int message_count_down = 30;
    while( true )
    {
        message_count_down--;
        if( message_count_down <= 0 )
        {
            res = false;
            break;
        }

        int num_bytes;
        if( sock.recv_msg( buffer, num_bytes, true ) ) //true = does redirect to new port address (differs from 6000)
        {
            buffer[ num_bytes ] = '\000';
        }
        else
        {
            break;
        }

        int type, time;
        res = SensorParser::get_message_type_and_time( buffer, type, time );

        if( !res )
            break;

        if( type == MESSAGE_SERVER_PARAM )
        {
            res = ServerParam::incorporate_server_param_string( buffer );
        }
        else if( type == MESSAGE_PLAYER_PARAM )
        {
            res = ServerParam::incorporate_player_param_string( buffer );
        }
        else if( type == MESSAGE_PLAYER_TYPE )
        {
            res = ServerParam::incorporate_player_type_string( buffer );
        }
        else if( type == MESSAGE_ERROR )
        {
            ERROR_OUT << ID << "\nreceived error message: " << buffer << flush;
            res = false;
        }
        else
        {
            ERROR_OUT << ID << "\nunknown message: " << buffer << flush;
            res = false;
        }

        if( !res )
            break;

        if( ServerParam::all_params_ok() )
            break;
    }
    return res;
}

/*
 * The following method just prints the information the client get from the
 * server on the outstream.
 */
bool Communicator::produce_parameter_messages( ostream &out )
{
    //the socket should be blocking in this routine
    bool res = false;
    int  number_of_player_types =  0;
    int  possible_player_types  =  0;
    int  message_count_down     = 30;

    out << "/*"
            << "\n  The following server parameters were automatically generated by"
            << "\n\n  bool produce_parameter_messages( UDPsocket &sock, ostream &out )"
            << "\n\n  in robocup/agent/src/client.cpp"
            << "\n*/";

    while( true )
    {
        message_count_down--;
        if( message_count_down <= 0 )
        {
            res = false;
            break;
        }

        int num_bytes;
        if( sock.recv_msg( buffer, num_bytes, true ) ) //true = does redirect to new port address (differs from 6000)
        {
            buffer[ num_bytes ] = '\000';
        }
        else
        {
            break;
        }

        int type, time;
        res = SensorParser::get_message_type_and_time( buffer, type, time );

        if( !res )
            break;

        if( type == MESSAGE_SERVER_PARAM )
        {
            if( res )
            {
                char const *dum = buffer;
                out << "\n\nconst char DEFAULT_MESSAGE_SERVER_PARAM[]=\"";
                while( *dum != '\0' )
                {
                    if( *dum == '"' )
                        out << '\\';
                    out << *dum;
                    dum++;
                }
                out << "\";";
            }
        }
        else if( type == MESSAGE_PLAYER_PARAM )
        {
            const char *str = buffer;
            out << "\n\nconst char DEFAULT_MESSAGE_PLAYER_PARAM[]=\"" << buffer << "\";";

            res = strskip(str, "(player_param "     , str ) &&
                    strfind(str, "(player_types"      , str ) &&
                    str2val(str, possible_player_types, str ) &&
                    strskip(str,')',str);

            if( res )
            {
                out << "\n\nconst int DEFAULT_NUM_MESSAGE_PLAYER_TYPE= " << possible_player_types << ";"
                        << "\nconst char * const DEFAULT_MESSAGE_PLAYER_TYPE[DEFAULT_NUM_MESSAGE_PLAYER_TYPE]= { ";
            }
        }
        else if( type == MESSAGE_PLAYER_TYPE )
        {
            number_of_player_types++;
            if( res )
            {
                if( number_of_player_types != 1 )
                    out << " ,\n      ";
                else
                    out << "\n      ";

                out << "\"" << buffer << "\"";
                if(number_of_player_types == possible_player_types )
                    out << "};";
            }
        }
        else if( type == MESSAGE_ERROR )
        {
            ERROR_OUT << ID << "\nreceived error message: " << buffer << flush;
            res = false;
        }
        else
        {
            ERROR_OUT << ID << "\nunknown message: " << buffer << flush;
            res = false;
        }
        if( !res )
            break;

        if( ServerParam::all_params_ok() )
            break;
    }

    return res;
}

/**
 * Converts a player's angle value of a command to the format requested
 * by the soccer server.
 *
 * @a angle to be converted
 * @return angle value being send to the soccer server
 */
Angle Communicator::ang_LP_2_SRV_deg( Angle a )
{
       //if( a < 0 || a > 2 * PI ) cerr << "\a= " << a;
       ANGLE tmp( a ); //normalize in 0..2PI
       a = tmp.get_value();

       if( a == PI )
       {
               WARNING_OUT << " an angle with value PI was set, this is not defined (see PI_MINUS_EPS)";
       }

       if( a < PI ) return - a * 180.0 / PI;
       return -( a - 2 * PI) * 180.0 / PI;
}

/**
 * Converts a player's x value of a command to the format requested
 * by the soccer server.
 */
double Communicator::x_LP_2_SRV( double x )
{
       return x;
}

/**
 * Converts a player's y value of a command to the format requested
 * by the soccer server.
 */
double Communicator::y_LP_2_SRV( double y )
{
       return -y;
}


void Communicator::send_cmd( Cmd const &cmd, Msg_teamcomm2 const &tc )
{
    /// new servers support more the one command in a message (say must be last!!!)

    ostringstream oss;

    double par1, par2;
    Angle ang;

    /***************************************************************************/
    //interpret main command, i.e. one of {moveto,turn,dash,kick,catch}
    if( cmd.cmd_body.is_cmd_set() )
    {
        switch( cmd.cmd_body.get_type() )
        {

        case Cmd_Body::TYPE_MOVETO   :
            cmd.cmd_body.get_moveto( par1, par2 );
            oss << "(move " << x_LP_2_SRV(par1) << " " << y_LP_2_SRV(par2) << ")";
            LOG_WM( WSinfo::ws->time, 0, "SENDING MOVE " << oss.str() );
            break;

        case Cmd_Body::TYPE_TURN     :
            cmd.cmd_body.get_turn( ang );
            oss << "(turn " << ang_LP_2_SRV_deg( ang ) << ")";
            LOG_WM( WSinfo::ws->time, 0, "SENDING TURN " << oss.str() );
            break;

        case Cmd_Body::TYPE_DASH     :
            cmd.cmd_body.get_dash( par1, ang );
            if( ClientOptions::server_version >= 13.0 )
                oss << "(dash " << par1 << " " << ang_LP_2_SRV_deg( ang ) << ")";
            else
                oss << "(dash " << par1 << ")";
            break;

        case Cmd_Body::TYPE_KICK     :
            cmd.cmd_body.get_kick( par1, ang );
            oss << "(kick " << par1 << " " << ang_LP_2_SRV_deg( ang ) << ")";
            break;

        case Cmd_Body::TYPE_CATCH    :
            cmd.cmd_body.get_catch( ang );
            oss << "(catch " << ang_LP_2_SRV_deg( ang ) << ")";
            break;

        case Cmd_Body::TYPE_TACKLE   :
            cmd.cmd_body.get_tackle( par1, par2 );
            if( ClientOptions::server_version >= 14.0 )
                oss << "(tackle " << par1 << " " << ( ( par2 > 0 ) ? "true" : "false" ) <<  ")";
            else
                oss << "(tackle " << par1 << ")";
            break;

        default:
            ERROR_OUT << ID << "\nwrong command";
        }
    }
    else
    { //es wurde kein Kommando gesetzt, evtl. warnen!!!
        //oss << "(turn 0)"; //als debug info verwendbar
        //ERROR_OUT << ID << "\nno command was specified, sending (turn 0)";
    }

    /***************************************************************************/
    // interpret pointto command
    if( cmd.cmd_point.is_cmd_set() )
    {
        double dist = 0;
        Angle ang  = 0;

        if ( !cmd.cmd_point.get_pointto_off() )
        {
            cmd.cmd_point.get_angle_and_dist( ang, dist );
            LOG_POL( 5, "JTS sending cmd_pointto got " << ang << " transformed to " << -RAD2DEG( Tools::get_angle_between_mPI_pPI( ang ) ) ); //<< ang_LP_2_SRV_deg( ang ) << endl);
            LOG_POL( 5, "JTS real neck_ang " << WSinfo::me->neck_ang );
            oss << "(pointto " << dist << " " << -RAD2DEG( Tools::get_angle_between_mPI_pPI( ang ) ) <<")";
        }
        else
        {
            oss << "(pointto off)";
        }
    }

    /***************************************************************************/
    //interpret neck command
    if( cmd.cmd_neck.is_cmd_set() )
    {
        cmd.cmd_neck.get_turn( ang );
        oss << "(turn_neck " << ang_LP_2_SRV_deg( ang ) << ")";
    }

    /***************************************************************************/
    //interpret view command
    if( cmd.cmd_view.is_cmd_set() )
    {
        if( WMoptions::behave_after_think )
        {
            /* this is a hack, because out change view strategy doesn't function when the cycle length is != 100ms */
            oss << "(change_view narrow high)";
        }
        else
        {
            int view_angle, view_quality;
            cmd.cmd_view.get_angle_and_quality(view_angle,view_quality);

            oss << "(change_view ";

            if( cmd.cmd_view.VIEW_ANGLE_WIDE == view_angle )
                oss << "wide ";
            else if( cmd.cmd_view.VIEW_ANGLE_NORMAL == view_angle )
                oss << "normal ";
            else
                oss << "narrow ";

            if ( cmd.cmd_view.VIEW_QUALITY_HIGH == view_quality)
                oss << "high)";
            else
                oss << "low)";
        }
    }


    //sending of say

    if( tc.get_num_objects() > 0 )
    {
        char *dum;
        char str[20]; //20 is enough for tc version 2
        bool res = SensorParser::manual_encode_teamcomm( str, tc, dum );

        if( !res || dum == str )
        {
            if( WSinfo::ws->time > 0 ) // Sput: get rid of err msg before game has started... */
            {
                ERROR_OUT << ID << "something wrong with teamcomm encoding\n" << tc;
            }
        }
        else
        {
            dum[ 0 ] = '\0';
            oss << "(say \"" << str << "\")";
        }
    }

/******************************************************************************/
/* in server 9.0.3 there is a bug, where nothing can come after an (attentionto off) */
/******************************************************************************/
    //interpret attention command
    //if ( WMoptions::behave_after_think )  //DEBUG, never leave it in normal code!!! (synch_mode seems not to work with attentionto)
    if( cmd.cmd_att.is_cmd_set() )
    {
        int p = -1;
        cmd.cmd_att.get_attentionto( p );
        if( p < 0 )
            oss << "(attentionto off)";
        else if( p > 0 && p <= NUM_PLAYERS )
            oss << "(attentionto our " << p << ")";
        else
        {
            oss << "(attentionto off)"; //will not do any harm and may help in this undefined situation
            ERROR_OUT << ID << "\nwrong attentionto parameter " << p;
        }
    }

    if ( WMoptions::behave_after_think )
        oss << "(done)";

    oss << ends;

    string buf = oss.str();

    if (buf.length() > 1)
    {
        sock.send_msg( buf.c_str(), buf.length() );
    }
}

#ifdef CMD_VIEW_HACK
void Communicator::send_cmd_view(const Cmd_View & cmd_view)
{
    ostringstream oss_view;

    if ( cmd_view.is_cmd_set() )
    {
        oss_view << "(change_view ";

        int va, vq;
        cmd_view.get_angle_and_quality( va, vq );

        switch( va )
        {
            case Cmd_View::VIEW_ANGLE_WIDE:   oss_view << "wide";   break;
            case Cmd_View::VIEW_ANGLE_NORMAL: oss_view << "normal"; break;
            case Cmd_View::VIEW_ANGLE_NARROW: oss_view << "narrow"; break;
        }

        if( vq == cmd_view.VIEW_QUALITY_HIGH ) { oss_view << " high)" << ends; }
        else                                   { oss_view << " low)"  << ends; }
    }

    string cmd_view_string = oss_view.str();

    if (cmd_view_string.length() > 0)
        sock.send_msg( cmd_view_string );
}
#endif

bool Communicator::send_string_directly( string str )
{
    return sock.send_msg( str );
}
