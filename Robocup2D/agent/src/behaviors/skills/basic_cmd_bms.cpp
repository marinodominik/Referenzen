#include "basic_cmd_bms.h"

#define BASELEVEL 3

bool BasicCmd::initialized = false;

BasicCmd::BasicCmd() {
    type_set = priority = type = -1;
    power = moment = 0;
    foul = false;
}

BasicCmd::~BasicCmd() {
}


bool BasicCmd::init (char const *conf_file, int argc, char const *const *argv)
{
    if (initialized) { return initialized; }

    initialized = true;

    if (! initialized)
    {
        std::cout << "\nBasicCmd behavior initialized.";
    }
    return initialized;
}


void BasicCmd::set_kick (double power, ANGLE direction)
{
    this->power     = power;
    this->direction = direction;

    this->type      = KICK;
    this->type_set  = WSinfo::ws->time;
}


void BasicCmd::set_turn (double moment)
{
    this->moment   = moment;

    this->type     = TURN;
    this->type_set = WSinfo::ws->time;
}

void BasicCmd::set_turn_inertia (double moment, Vector const & vel)
{
    this->moment   = moment;
    this->vel      = vel;

    this->type     = TURN_INERTIA;
    this->type_set = WSinfo::ws->time;
}

void BasicCmd::set_turn_inertia (double moment)
{
    set_turn_inertia (moment, WSinfo::me->vel);
}


void BasicCmd::set_dash (double power, int priority)
{
    this->power    = power;
    this->priority = priority;

    this->type     = DASH;
    this->type_set = WSinfo::ws->time;
}

void BasicCmd::set_dash (double power, ANGLE direction)
{
    this->power     = power;
    this->direction = direction;

    this->type      = SIDE_DASH;
    this->type_set  = WSinfo::ws->time;
}

void BasicCmd::set_tackle (double power, bool foul)
{
    this->power    = power;
    this->foul     = foul;

    this->type     = TACKLE;
    this->type_set = WSinfo::ws->time;
}

void BasicCmd::set_tackle (ANGLE const &angle, bool foul)
{
    set_tackle (angle.get_value_mPI_pPI () * 180. / PI, foul);
}


void BasicCmd::set_catch (ANGLE direction)
{
    this->direction = direction;

    this->type      = CATCH;
    this->type_set  = WSinfo::ws->time;
}


void BasicCmd::set_move (Vector pos)
{
    this->pos      = pos;

    this->type     = MOVE;
    this->type_set = WSinfo::ws->time;
}


bool BasicCmd::get_cmd (Cmd &cmd)
{
    if( type_set != WSinfo::ws->time ) {
        ERROR_OUT << "BasicCmd: Type not set before calling get_cmd()!";
        return false;
    }

    switch (type) {
    case KICK:
        cmd.cmd_body.set_kick (power, direction.get_value_mPI_pPI ());
        break;
    case TURN:
        cmd.cmd_body.set_turn (moment);
        break;
    case TURN_INERTIA:
        do_turn_inertia (cmd.cmd_body, moment, vel);
        break;
    case DASH:
        cmd.cmd_body.set_dash (power, priority);
        break;
    case SIDE_DASH:
        cmd.cmd_body.set_dash (power, direction.get_value_mPI_pPI ());
        break;
    case TACKLE:
        cmd.cmd_body.set_tackle (power, foul);
        break;
    case CATCH:
        cmd.cmd_body.set_catch (direction.get_value_mPI_pPI ());
        break;
    case MOVE:
        cmd.cmd_body.set_moveto (pos.getX (), pos.getY ());
        break;
    default:
        ERROR_OUT << "BasicCmd: Unknown command type!";
        return false;
    }

    return true;
}

void BasicCmd::do_turn_inertia (Cmd_Body &cmd, double moment, Vector const & vel) {
    moment = Tools::get_angle_between_mPI_pPI (moment);
    moment = moment * (1.0 + (WSinfo::me->inertia_moment * (vel.norm())));

    if (moment >  3.14159) moment =  3.14159;
    if (moment < -3.14159) moment = -3.14159;

    moment = Tools::get_angle_between_null_2PI (moment);
    cmd.set_turn (moment);
}