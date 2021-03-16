#include "Cmd.h"

#include <string.h>
#include "ws_info.h"


// |----------------------------------------------------------------|
// |  ,-----.             ,--.      ,-----.                         |
// | '  .--./,--,--,--. ,-|  |      |  |) /_  ,--,--. ,---.  ,---.  |
// | |  |    |        |' .-. |      |  .-.  \' ,-.  |(  .-' | .-. : |
// | '  '--'\|  |  |  |\ `-' |,----.|  '--' /\ '-'  |.-'  `)\   --. |
// |  `-----'`--`--`--' `---' '----'`------'  `--`--'`----'  `----' |
// |                                                                |
// |----------------------------------------------------------------|
Cmd_Base::Cmd_Base()
{
	lock_set = false;
	cmd_set  = false;
}

bool Cmd_Base::set_lock()
{
    if( check_cmd() )
    {
        bool lock_is_already_set = check_lock();
        if( !lock_is_already_set )
        {
            lock_set = true;
        }
        return !lock_is_already_set;
    }
    else
    {
        return false;
    }
}

bool Cmd_Base::unset_lock()
{
    bool lock_is_already_unset = !is_lock_set();
    if( !lock_is_already_unset )
    {
        lock_set = false;
    }
    return !lock_is_already_unset;
}

bool Cmd_Base::is_lock_set() const
{
    return lock_set;
}

bool Cmd_Base::check_lock() const
{
    bool lock_is_set = is_lock_set();
    if( lock_is_set )
    {
        ERROR_OUT << "\n this command is locked";
    }
    return lock_is_set;
}

bool Cmd_Base::set_cmd()
{
    if( !check_lock() )
    {
        bool cmd_is_already_set = check_cmd();
        if( !cmd_is_already_set )
        {
            cmd_set = true;
        }
        return !cmd_is_already_set;
    }
    else{
        return false;
    }
}

bool Cmd_Base::unset_cmd()
{
    if( !check_lock() )
    {
        bool cmd_is_already_unset = !is_cmd_set();
        if( !cmd_is_already_unset )
        {
            cmd_set = false;
        }
        return !cmd_is_already_unset;
    }
    else{
        return false;
    }
}

bool Cmd_Base::is_cmd_set() const
{
    return cmd_set;
}

bool Cmd_Base::check_cmd() const
{
    bool cmd_is_set = is_cmd_set();
    if( !cmd_is_set )
    {
        ERROR_OUT << "\n no command was set";
    }
    return cmd_is_set;
}



// |-----------------------------------------------------------------|
// |  ,-----.             ,--.      ,-----.            ,--.          |
// | '  .--./,--,--,--. ,-|  |      |  |) /_  ,---.  ,-|  |,--. ,--. |
// | |  |    |        |' .-. |      |  .-.  \| .-. |' .-. | \  '  /  |
// | '  '--'\|  |  |  |\ `-' |,----.|  '--' /' '-' '\ `-' |  \   '   |
// |  `-----'`--`--`--' `---' '----'`------'  `---'  `---' .-'  /    |
// |                                                       `---'     |
// |-----------------------------------------------------------------|
void Cmd_Body::check_type( int t ) const
{
    check_cmd();
    if( type != t ) ERROR_OUT << "wrong type";
}

Cmd_Body::Cmd_Body():Cmd_Base()
{
    type      = TYPE_NONE;
    param_1   = 0.0;
    param_2   = 0.0;
    priority  = 0;
}

bool Cmd_Body::clone( Cmd_Body const &cmd )
{
    if( is_lock_set() )
    {
        unset_lock();
    }
    cmd.is_cmd_set()  ? set_cmd()  : unset_cmd() ;
    cmd.is_lock_set() ? set_lock() : unset_lock();
    type      = cmd.type;
    param_1   = cmd.param_1;
    param_2   = cmd.param_2;
    param_ang = cmd.param_ang;
    priority  = cmd.priority;
    return true;
}

int Cmd_Body::get_type() const
{
    return type;
}

bool Cmd_Body::is_type( int type )
{
    return ( this->type == type );
}

int Cmd_Body::get_type( double &power, Angle &angle )
{
    power = param_1;
    angle = param_ang.get_value_0_p2PI();
    return type;
}

int Cmd_Body::get_type( double &power, ANGLE &angle )
{
    power = param_1;
    angle = param_ang;
    return type;
}

int Cmd_Body::get_priority() const
{
    return priority;
}

void Cmd_Body::set_none()
{
    check_lock();
    set_lock();
    set_cmd();
}

void Cmd_Body::set_moveto( double x, double y )
{
    check_lock();
    set_lock();
    set_cmd();
    type     = TYPE_MOVETO;
    param_1  = x;
    param_2  = y;
    priority = 0;
}

void Cmd_Body::get_moveto( double &x, double &y ) const
{
    check_type( TYPE_MOVETO );
    x = param_1;
    y = param_2;
}

void Cmd_Body::set_dash( double power, int p_priority )
{
    check_lock();
    set_lock();
    set_cmd();
    type      = TYPE_DASH;
    param_1   = power;
    param_ang = ANGLE();
    priority  = p_priority;
}

void Cmd_Body::set_dash( double power, Angle angle, int p_priority )
{
    check_lock();
    set_lock();
    set_cmd();
    type      = TYPE_DASH;
    param_1   = power;
    param_ang = ANGLE(angle);
    priority  = p_priority;
}

void Cmd_Body::set_dash( double power, ANGLE angle, int p_priority )
{
    check_lock();
    set_lock();
    set_cmd();
    type      = TYPE_DASH;
    param_1   = power;
    param_ang = angle;
    priority  = p_priority;
}

void Cmd_Body::get_dash( double &power ) const
{
    check_type( TYPE_DASH );
    power = param_1;
}

void Cmd_Body::get_dash( double &power, Angle &angle ) const
{
    check_type( TYPE_DASH );
    power = param_1;
    angle = param_ang.get_value_0_p2PI();
}

void Cmd_Body::get_dash( double &power, ANGLE &angle ) const
{
    check_type( TYPE_DASH );
    power = param_1;
    angle = param_ang;
}

void Cmd_Body::set_turn( Angle angle )
{
    check_lock();
    set_lock();
    set_cmd();
    type      = TYPE_TURN;
    param_ang = ANGLE(angle);
    priority  = 0;
}

void Cmd_Body::set_turn( ANGLE const &angle )
{
    check_lock();
    set_lock();
    set_cmd();
    type      = TYPE_TURN;
    param_ang = angle;
    priority  = 0;
}

void Cmd_Body::get_turn( Angle &angle ) const
{
    check_type( TYPE_TURN );
    angle = param_ang.get_value_0_p2PI();
}

void Cmd_Body::get_turn( ANGLE &angle ) const
{
    check_type( TYPE_TURN );
    angle = param_ang;
}

void Cmd_Body::set_tackle( double angle )
{
    set_tackle( angle, false );
}

void Cmd_Body::set_tackle( double angle, bool foul )
{
    check_lock();
    set_lock();
    set_cmd();
    type      = TYPE_TACKLE;
    param_ang = ANGLE( angle );
    param_1   = foul;
    priority  = 0;
}

void Cmd_Body::set_tackle( ANGLE angle, bool foul )
{
    check_lock();
    set_lock();
    set_cmd();
    type      = TYPE_TACKLE;
    param_ang = angle;
    param_1   = foul;
    priority  = 0;
}

void Cmd_Body::get_tackle( double &angle, double &foul ) const
{
    check_type( TYPE_TACKLE );
    angle = param_ang.get_value();
    foul  = param_1;
}

void Cmd_Body::get_tackle( ANGLE &angle, double &foul ) const
{
    check_type( TYPE_TACKLE );
    angle = param_ang;
    foul  = param_1;
}

void Cmd_Body::set_kick( double power, Angle angle )
{
    check_lock();
    set_lock();
    set_cmd();
    type      = TYPE_KICK;
    param_1   = power;
    param_ang = ANGLE(angle);
    priority  = 0;
}

void Cmd_Body::set_kick( double power, ANGLE angle )
{
    check_lock();
    set_lock();
    set_cmd();
    type      = TYPE_KICK;
    param_1   = power;
    param_ang = angle;
    priority  = 0;
}

void Cmd_Body::get_kick( double &power, Angle &angle ) const
{
    check_type( TYPE_KICK );
    power = param_1;
    angle = param_ang.get_value_0_p2PI();
}

void Cmd_Body::get_kick( double &power, ANGLE &angle ) const
{
    check_type( TYPE_KICK );
    power = param_1;
    angle = param_ang;
}

void Cmd_Body::set_catch( Angle angle )
{
    check_lock();
    set_lock();
    set_cmd();
    type      = TYPE_CATCH;
    param_ang = ANGLE(angle);
    priority  = 0;
}

void Cmd_Body::set_catch( ANGLE angle )
{
    check_lock();
    set_lock();
    set_cmd();
    type      = TYPE_CATCH;
    param_ang = angle;
    priority  = 0;
}

void Cmd_Body::get_catch( Angle &angle ) const
{
    check_type( TYPE_CATCH );
    angle = param_ang.get_value_0_p2PI();
}

void Cmd_Body::get_catch( ANGLE &angle ) const
{
    check_type( TYPE_CATCH );
    angle = param_ang;
}

void Cmd_Body::check_lock() const
{
    if( is_lock_set() ) ERROR_OUT << "\n time: " << WSinfo::ws->time << " player: " << WSinfo::me->number << "this command is locked, type " << get_type() << "\n";
}

std::ostream& operator<<( std::ostream &o, const Cmd_Body &cmd )
{
    double param_1, param_2;
    Angle param_ang;
#if 0
    o << "\n [ ";
    if( cmd.is_lock_set() )
        o << " <locked>, ";
#endif
    if( !cmd.is_cmd_set() )
        o << "(no command)";
    else
        switch( cmd.type ) {
        case Cmd_Body::TYPE_MOVETO:
            cmd.get_moveto( param_1, param_2 );
            o << "(move " << param_1 << " " << param_2 << ")";
            break;
        case Cmd_Body::TYPE_TURN:
            cmd.get_turn( param_ang );
            o << "(turn " << param_ang << ")";
            break;
        case Cmd_Body::TYPE_DASH:
            cmd.get_dash( param_1, param_ang );
            o << "(dash " << param_1 << " " << param_ang << ")";
            break;
        case Cmd_Body::TYPE_KICK:
            cmd.get_kick( param_1, param_ang );
            o << "(kick " << param_1 << " " << param_ang << ")";
            break;
        case Cmd_Body::TYPE_CATCH:
            cmd.get_catch( param_ang );
            o << "(catch " << param_ang << ")";
            break;
        case Cmd_Body::TYPE_TACKLE:
            cmd.get_tackle( param_1, param_2 );
            o << "(tackle " << param_1 << " " << param_2 << ")";
            break;
        default:
            o << "(none)";
        }
#if 0
    o << " ]";
#endif
    return o;
}

void Cmd_Body::reset()
{
    type      = TYPE_NONE;
    param_1   = 0.0;
    param_2   = 0.0;
    priority  = 0;

    if( is_lock_set() )
    {
        unset_lock();
    }

    if( is_cmd_set() )
    {
        unset_cmd();
    }
}

// |----------------------------------------------------------------|
// |  ,-----.             ,--.      ,--.  ,--.             ,--.     |
// | '  .--./,--,--,--. ,-|  |      |  ,'.|  | ,---.  ,---.|  |,-.  |
// | |  |    |        |' .-. |      |  |' '  || .-. :| .--'|     /  |
// | '  '--'\|  |  |  |\ `-' |,----.|  | `   |\   --.\ `--.|  \  \  |
// |  `-----'`--`--`--' `---' '----'`--'  `--' `----' `---'`--'`--' |
// |                                                                |
// |----------------------------------------------------------------|
Cmd_Neck::Cmd_Neck():Cmd_Base()
{
    param_ang = 0.0;
}

void Cmd_Neck::set_turn( Angle angle )
{
    check_lock();
    set_lock();
    set_cmd();
    param_ang = ANGLE(angle);
}

void Cmd_Neck::set_turn( ANGLE angle )
{
    check_lock();
    set_lock();
    set_cmd();
    param_ang = angle;
}

void Cmd_Neck::get_turn( Angle &angle ) const
{
    check_cmd();
    angle = param_ang.get_value_0_p2PI();
}

void Cmd_Neck::get_turn( ANGLE &angle ) const
{
    check_cmd();
    angle = param_ang;
}

std::ostream& operator<<( std::ostream &o, const Cmd_Neck &cmd )
{
    Angle ang;
    o << "\n [ ";
    if( cmd.is_lock_set() )
        o << " <locked>, ";
    if( !cmd.is_cmd_set() )
        o << " (no command)";
    else
    {
        cmd.get_turn( ang );
        o << " (turn_neck " << ang << ")";
    }
    o << " ]";
    return o;
}

void Cmd_Neck::reset()
{
    param_ang = 0.0;

    if( is_lock_set() )
    {
        unset_lock();
    }

    if( is_cmd_set() )
    {
        unset_cmd();
    }
}

// |---------------------------------------------------------------|
// |  ,-----.             ,--.   ,--.   ,--.,--.                   |
// | '  .--./,--,--,--. ,-|  |    \  `.'  / `--' ,---. ,--.   ,--. |
// | |  |    |        |' .-. |     \     /  ,--.| .-. :|  |.'.|  | |
// | '  '--'\|  |  |  |\ `-' |,----.\   /   |  |\   --.|   .'.   | |
// |  `-----'`--`--`--' `---' '----' `-'    `--' `----''--'   '--' |
// |                                                               |
// |---------------------------------------------------------------|
Cmd_View::Cmd_View():Cmd_Base()
{
    view_angle   = VIEW_ANGLE_NARROW;
    view_quality = VIEW_QUALITY_LOW;
}

void Cmd_View::set_angle_and_quality( int ang, int quality )
{
    check_lock();
    set_lock();
    set_cmd();
    view_angle   = ang;
    view_quality = quality;
}

void Cmd_View::get_angle_and_quality( int &ang, int &quality ) const
{
    check_cmd();
    ang     = view_angle;
    quality = view_quality;
}

void Cmd_View::check_lock() const
{
    if( is_lock_set() ) ERROR_OUT << "\n this command is locked, type view";
}

std::ostream& operator<<( std::ostream &o, const Cmd_View &cmd )
{
    int angle, quality;
    o << "\n [ ";
    if( cmd.is_lock_set() )
        o << " <locked>, ";
    if( !cmd.is_cmd_set() )
        o << " (no command)";
    else
    {
        cmd.get_angle_and_quality( angle, quality );
        o << " (change_view ";
        switch( angle )
        {
        case Cmd_View::VIEW_ANGLE_WIDE   :
            o << "wide ";
            break;
        case Cmd_View::VIEW_ANGLE_NORMAL :
            o << "normal ";
            break;
        case Cmd_View::VIEW_ANGLE_NARROW :
            o << "narrow ";
            break;
        default: 
            o << " ? ";
        };
        switch( quality )
        {
        case Cmd_View::VIEW_QUALITY_HIGH :
            o << "high)";
            break;
        case Cmd_View::VIEW_QUALITY_LOW  :
            o << "low)";
            break;
        default:
            o << " ? ";
        }
    }
    o << " ]";
    return o;
}

void Cmd_View::reset()
{
    view_angle   = VIEW_ANGLE_NARROW;
    view_quality = VIEW_QUALITY_LOW;

    if( is_lock_set() )
    {
        unset_lock();
    }

    if( is_cmd_set() )
    {
        unset_cmd();
    }
}

// |---------------------------------------------------------------------|
// |  ,-----.             ,--.      ,------.        ,--.          ,--.   |
// | '  .--./,--,--,--. ,-|  |      |  .--. ' ,---. `--',--,--, ,-'  '-. |
// | |  |    |        |' .-. |      |  '--' || .-. |,--.|      \'-.  .-' |
// | '  '--'\|  |  |  |\ `-' |,----.|  | --' ' '-' '|  ||  ||  |  |  |   |
// |  `-----'`--`--`--' `---' '----'`--'      `---' `--'`--''--'  `--'   |
// |                                                                     |
// |---------------------------------------------------------------------|
Cmd_Point::Cmd_Point():Cmd_Base()
{
    off_wanted = false;
    angle      = -1.0;
    dist       = -1.0;
}

void Cmd_Point::set_pointto( double distance, Angle ang )
{
    check_lock();
    set_lock();
    set_cmd();
    off_wanted = false;
    // compute dist and angle
    dist       = distance;
    angle      = ANGLE(ang);
}

void Cmd_Point::set_pointto( double distance, ANGLE ang )
{
    check_lock();
    set_lock();
    set_cmd();
    off_wanted = false;
    // compute dist and angle
    dist       = distance;
    angle      = ang;
}

void Cmd_Point::set_pointto_off()
{
    check_lock();
    set_lock();
    set_cmd();
    off_wanted = true;
}

bool Cmd_Point::get_pointto_off() const
{
    return off_wanted;
}

void Cmd_Point::get_angle_and_dist( Angle &ang, double &distance ) const
{
    check_cmd();
    ang      = angle.get_value_0_p2PI();
    distance = dist;
}

void Cmd_Point::get_angle_and_dist( ANGLE &ang, double &distance ) const
{
    check_cmd();
    ang      = angle;
    distance = dist;
}

void Cmd_Point::check_lock() const
{
    if( is_lock_set() ) ERROR_OUT << "\n this command is locked, type pointto";
}

std::ostream& operator<<( std::ostream &o, const Cmd_Point &cmd )
{
    Angle ang;
    double distance;
    o << "\n [ ";
    if( cmd.is_lock_set() )
        o << " <locked>, ";
    if( !cmd.is_cmd_set() )
        o << " (no command)";
    else
    {
        if( !cmd.get_pointto_off() )
        {
            cmd.get_angle_and_dist( ang, distance );
            o << " (pointto " << distance << " " << ang << ")";
        }
        else
        {
            o << " (pointto off)";
        }
    }
    o << " ]";
    return o;
}

void Cmd_Point::reset()
{
    off_wanted = false;
    angle      = -1.0;
    dist       = -1.0;

    if( is_lock_set() )
    {
        unset_lock();
    }

    if( is_cmd_set() )
    {
        unset_cmd();
    }
}

// |----------------------------------------------------------|
// |  ,-----.             ,--.       ,---.                    |
// | '  .--./,--,--,--. ,-|  |      '   .-'  ,--,--.,--. ,--. |
// | |  |    |        |' .-. |      `.  `-. ' ,-.  | \  '  /  |
// | '  '--'\|  |  |  |\ `-' |,----..-'    |\ '-'  |  \   '   |
// |  `-----'`--`--`--' `---' '----'`-----'  `--`--'.-'  /    |
// |                                                `---'     |
// |----------------------------------------------------------|
Cmd_Say::Cmd_Say():Cmd_Base()
{
    pass.valid                       = false;
    ball.valid                       = false;
    ball_holder.valid                = false;
    players.num                      = 0;
    direct_opponent_assignment.valid = false; //TGdoa
    pass_request.valid               = false; //TGpr
    msg.valid                        = false;
}

void Cmd_Say::check_lock() const
{
    if( is_lock_set() ) ERROR_OUT << "\n this command is locked, type say";
}

void Cmd_Say::set_pass( Vector const &pos, Vector const &vel, int time )
{
    check_lock();
    set_lock();
    set_cmd();
    pass.valid    = true;
    pass.ball_pos = pos;
    pass.ball_vel = vel;
    pass.time     = time;
}

bool Cmd_Say::get_pass( Vector &pos, Vector &vel, int &time ) const
{
    if( !pass.valid )
        return false;

    pos  = pass.ball_pos;
    vel  = pass.ball_vel;
    time = pass.time;
    return true;
}

bool Cmd_Say::pass_valid() const
{
    return pass.valid;
}

void Cmd_Say::set_me_as_ball_holder( Vector const &pos )
{
    check_lock();
    set_lock();
    set_cmd();
    ball_holder.valid = true;
    ball_holder.pos   = pos;
}

bool Cmd_Say::get_ball_holder( Vector &pos ) const
{
    if( !ball_holder.valid )
        return false;

    pos = ball_holder.pos;
    return true;
}

bool Cmd_Say::ball_holder_valid() const
{
    return ball_holder.valid;
}

void Cmd_Say::set_ball( Vector const &pos, Vector const &vel, int age_pos, int age_vel )
{
    check_lock();
    set_lock();
    set_cmd();
    ball.valid    = true;
    ball.ball_pos = pos;
    ball.ball_vel = vel;
    ball.age_pos  = age_pos;
    ball.age_vel  = age_vel;
}

bool Cmd_Say::get_ball( Vector &pos, Vector &vel, int &age_pos, int &age_vel ) const
{
    if( !ball.valid )
        return false;

    pos     = ball.ball_pos;
    vel     = ball.ball_vel;
    age_pos = ball.age_pos;
    age_vel = ball.age_vel;
    return true;
}

bool Cmd_Say::ball_valid() const
{
    return ball.valid;
}

void Cmd_Say::set_players( PlayerSet const &pset )
{
    players.num = pset.num;
    if( players.num > players.max_num )
        players.num = players.max_num;

    for( int i = 0; i < players.num; i++ )
    {
        players.player[i].pos    = pset[i]->pos;
        players.player[i].team   = pset[i]->team;
        players.player[i].number = pset[i]->number;
    }
}

bool Cmd_Say::get_player( int idx, Vector &pos, int &team, int &number ) const
{
    if( idx >= players.num )
        return false;

    pos    = players.player[idx].pos;
    team   = players.player[idx].team;
    number = players.player[idx].number;
    return true;
}

int Cmd_Say::get_players_num() const
{
    return players.num;
}

//TGdoa: begin
void Cmd_Say::set_direct_opponent_assignment( int assgnmnt )
{
    check_lock();
    set_lock();
    set_cmd();
    direct_opponent_assignment.valid      = true;
    direct_opponent_assignment.assignment = assgnmnt;
}

bool Cmd_Say::get_direct_opponent_assignment( int &assgnmnt ) const
{
    if( !direct_opponent_assignment.valid )
        return false;

    assgnmnt = direct_opponent_assignment.assignment;
    return true;
}

bool Cmd_Say::direct_opponent_assignment_valid() const
{
    return direct_opponent_assignment.valid;
}
//TGdoa: end

//TGpr: begin
void Cmd_Say::set_pass_request( int pass_in_n_steps, int pass_param )
{
    check_lock();
    set_lock();
    set_cmd();
    pass_request.valid           = true;
    pass_request.pass_in_n_steps = pass_in_n_steps;
    pass_request.pass_param      = pass_param;
}

bool Cmd_Say::get_pass_request( int &pass_in_n_steps, int &pass_param ) const
{
    if( !pass_request.valid )
        return false;

    pass_in_n_steps = pass_request.pass_in_n_steps;
    pass_param      = pass_request.pass_param;
    return true;
}

bool Cmd_Say::pass_request_valid() const
{
    return pass_request.valid;
}
//TGpr: end

void Cmd_Say::set_msg( unsigned char type, short p1, short p2 )
{
    check_lock();
    set_lock();
    set_cmd();
    msg.valid  = true;
    msg.param1 = p1;
    msg.param2 = p2;
    msg.type   = type; // hauke
}

bool Cmd_Say::get_msg( unsigned char &type, short &p1, short &p2 ) const
{
    if( !msg.valid )
        return false;

    p1   = msg.param1;
    p2   = msg.param2;
    type = msg.type; // hauke
    return true;
}

bool Cmd_Say::get_msg( SayMsg &m) const
{
    m = msg;
    return true;
}

bool Cmd_Say::msg_valid() const
{
    return msg.valid;
}

std::ostream& operator<<( std::ostream &o, const Cmd_Say &cmd )
{
    o << "\n [ ";
    if( cmd.is_lock_set() )
        o << " <locked>, ";
    if( !cmd.is_cmd_set() )
        o << " (no command)";
    else
    {
        // o << " (say <...> )";// << cmd.message << ")";
        o << "pass.valid=" << cmd.pass.valid << " pass.ball_pos=" << cmd.pass.ball_pos << " pass.ball_vel=" << cmd.pass.ball_vel << " pass.time=" << cmd.pass.time << " ball_holder.valid=" << cmd.ball_holder.valid << " ball_holder.pos=" << cmd.ball_holder.pos << " ball.valid=" << cmd.ball.valid << " ball.ball_pos=" << cmd.ball.ball_pos << " ball.ball_vel=" << cmd.ball.ball_vel << " ball.age_pos=" << cmd.ball.age_pos << " ball.age_vel=" << cmd.ball.age_vel << " players.num=" << cmd.players.num << " doa.valid=" << cmd.direct_opponent_assignment.valid;
    }
    o << " ]";
    return o;
}

void Cmd_Say::reset()
{
    pass.valid                       = false;
    ball.valid                       = false;
    ball_holder.valid                = false;
    players.num                      = 0;
    direct_opponent_assignment.valid = false; //TGdoa
    pass_request.valid               = false; //TGpr
    msg.valid                        = false;

    if( is_lock_set() )
    {
        unset_lock();
    }

    if( is_cmd_set() )
    {
        unset_cmd();
    }
}

// |----------------------------------------------------------------------------------------------------|
// |  ,-----.             ,--.        ,---.    ,--.    ,--.                   ,--.  ,--.                |
// | '  .--./,--,--,--. ,-|  |       /  O  \ ,-'  '-.,-'  '-. ,---. ,--,--, ,-'  '-.`--' ,---. ,--,--,  |
// | |  |    |        |' .-. |      |  .-.  |'-.  .-''-.  .-'| .-. :|      \'-.  .-',--.| .-. ||      \ |
// | '  '--'\|  |  |  |\ `-' |,----.|  | |  |  |  |    |  |  \   --.|  ||  |  |  |  |  |' '-' '|  ||  | |
// |  `-----'`--`--`--' `---' '----'`--' `--'  `--'    `--'   `----'`--''--'  `--'  `--' `---' `--''--' |
// |                                                                                                    |
// |----------------------------------------------------------------------------------------------------|
Cmd_Attention::Cmd_Attention():Cmd_Base()
{
    player = -1;
}

void Cmd_Attention::set_attentionto_none()
{
    check_lock();
    set_lock();
    set_cmd();
    player   = -1;
}

void Cmd_Attention::set_attentionto( int p )
{
    check_lock();
    set_lock();
    set_cmd();
    player   = p;
}

void Cmd_Attention::get_attentionto( int &p ) const
{
    check_cmd();
    p = player;
}

void Cmd_Attention::check_lock() const
{
    if( is_lock_set() ) ERROR_OUT << "\n this command is locked, type attention";
}

void Cmd_Attention::reset()
{
    player = -1;

    if( is_lock_set() )
    {
        unset_lock();
    }

    if( is_cmd_set() )
    {
        unset_cmd();
    }
}

// |---------------------------|
// |  ,-----.             ,--. |
// | '  .--./,--,--,--. ,-|  | |
// | |  |    |        |' .-. | |
// | '  '--'\|  |  |  |\ `-' | |
// |  `-----'`--`--`--' `---'  |
// |                           |
// |---------------------------|
void Cmd::reset()
{
    cmd_body.reset();
    cmd_neck.reset();
    cmd_view.reset();
    cmd_say.reset();
    cmd_att.reset();
    cmd_point.reset();
}

std::ostream& operator<<( std::ostream &o, const Cmd &cmd )
{
    o << "\n--\nCmd:"
        << "\ncmd_main" << cmd.cmd_body
        << "\ncmd_neck" << cmd.cmd_neck
        << "\ncmd_view" << cmd.cmd_view
        << "\ncmd_say"  << cmd.cmd_say;
    return o;
}
