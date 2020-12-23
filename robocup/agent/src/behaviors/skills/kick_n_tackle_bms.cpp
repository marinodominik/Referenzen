//
// Created by Eicke Godehardt on 2/24/17.
//

#include "kick_n_tackle_bms.h"
#include "basics/log_macros.h"
#include "basics/tools.h"


bool KickNTackle::initialized = false;


KickNTackle::KickNTackle ()
{
    interceptball = new InterceptBall;
//    neuroKick     = new NeuroKickWrapper;
    basic         = new BasicCmd;
    kick          = new OneStepKick;

    last_simulation_time = -1;
    last_execution_time  = -1;
}


KickNTackle::~KickNTackle ()
{
    if (interceptball) delete interceptball;
//    if (neuroKick)     delete neuroKick;
    if (basic)         delete basic;
    if (kick)          delete kick;
}


bool KickNTackle::init (char const *conf_file, int argc, char const *const *argv)
{
    if (initialized) {
        return true;
    }

    bool returnValue = true;
    returnValue &= InterceptBall::init    (conf_file, argc, argv);
//    returnValue &= NeuroKickWrapper::init (conf_file, argc, argv);
    returnValue &= BasicCmd::init         (conf_file, argc, argv);
    returnValue &= OneStepKick::init      (conf_file, argc, argv);

    return returnValue;
}


bool KickNTackle::is_possible (const Vector &target, double success /* = .8 */, double min_ball_speed /* = 1.5 */)
{
    bool result = false;
    this->target = target;

//    printf ("is possible (%d, %d, %d)\n", last_simulation_time, last_execution_time, WSinfo::ws->time);
    if (last_execution_time + 1 < WSinfo::ws->time)
    { // from scratch
        last_probability = -1;
        if (WSinfo::is_ball_kickable ())
        {
            result = simulate_all (success, min_ball_speed);
        }
        else
        {
            // initially in tackle range without kicking before
            result = false;
            /*
            MyState initial = MyState ();
            MyState next    = MyState ();
            initial.get_from_WS ();
            Cmd cmd = Cmd ();
            Tools::get_successor_state (initial, cmd.cmd_body, next);

            if (next.my_pos.distance (next.ball_pos) <= 2.)
            { // can tackle
                // TODO implement
                // is turn possible in one turn?
                // calculate tackle possibility
            }
             */
        }
    }
    else
    { // somewhere in progress
        if (state == KICK)
        {
            result = simulate_turn_tackle (success, min_ball_speed);
        }
        else if (state == TURN)
        {
            result = simulate_tackle (success, min_ball_speed);
        }
    }
//    printf ("*** result of is_possible = %d\n", result);
    if (result)
    {
        last_simulation_time = WSinfo::ws->time;
    }
    return result;
}


bool KickNTackle::simulate_all (double min_success, double min_ball_speed)
{
    Cmd cmd = Cmd ();
    MyState initial    = MyState ();
    MyState next       = MyState ();
    MyState after_kick = MyState ();
    MyState after_turn = MyState ();
    MyState final      = MyState ();
    initial.get_from_WS ();
    next = initial; // target positions taken from start center point, but actual player still moves
    // uncomment next line in favor of the one above if you want more accurate circle around player
//    Tools::get_successor_state (initial, cmd.cmd_body, next);
    int max_angle = -1;

    for (int i = -70; i <= 70; i += 10)
    {
        double angle;
        double power;
        ANGLE tackle_angle = ANGLE (PI * i / 180);

//        printf ("----- %3d (%3d) -------------------------------------------\n", i, i < 0 ? i + 360 : i);
//        printf ("body direction=%.0f\t (tackle angle=%f)\n", initial.my_angle.get_value () / PI * 180, tackle_angle.get_value ());


        // === sim kick ============================================================================================
        // 2 time steps to reach target position
        const Vector &kick_target = next.my_pos + Vector (1.5, 0).ROTATE (tackle_angle);
        bool success = find_optimal_kick (initial, after_kick, kick_target, 2, angle, power);
        if (! success)
        {
//            printf ("unusable kick parameter!\n");
            LOG_POL (0, << _2D << L2D (initial.ball_pos.getX (), initial.ball_pos.getY (),
                    kick_target.getX (), kick_target.getY (), "000000"));
            LOG_POL (0, << _2D << STRING2D (kick_target.getX (), kick_target.getY (), "" << (i), "000000"));
            continue;
        }
//            printf ("after sim kick %f  %f\n", kick_angle.get_value (), kick_power);


        // === sim turn ============================================================================================
        // turning to angle as 2nd step
        double turn_diff = (Vector (tackle_angle).ARG () - after_kick.my_angle).get_value_mPI_pPI ();// / PI * 180.;
        basic->set_turn_inertia (turn_diff, after_kick.my_vel);
        basic->get_cmd (cmd);
        Tools::get_successor_state (after_kick, cmd.cmd_body, after_turn);


        // === sim tackle ==========================================================================================
//        double ball_dir = (after_turn.ball_pos - after_turn.my_pos).ARG ().get_value_mPI_pPI ();// / PI * 180.;
//        printf ("dist=%4.2f\t ball dir=%4.0f\n", dist, ball_dir / PI * 180);
//        printf ("my pos=(%4.1f,%4.1f)\t ball pos=(%4.1f,%4.1f)\t lot dist=%5.2f\n",
//                intermediate.my_pos.getX (),   intermediate.my_pos.getY (),
//                intermediate.ball_pos.getX (), intermediate.ball_pos.getY (),
//                Tools::get_dist2_line (intermediate.my_pos, intermediate.my_pos + Vector (intermediate.my_angle), intermediate.ball_pos));
        double final_tackle_angle = 4711;
        success = find_optimal_tackle (after_turn, final, final_tackle_angle);
        if (! success)
        {
//            printf ("unusable tackle parameter!\n");
            LOG_POL (0, << _2D << L2D (initial.ball_pos.getX (), initial.ball_pos.getY (),
                    after_turn.ball_pos.getX (), after_turn.ball_pos.getY (), "5555ff"));
            LOG_POL (0, << _2D << STRING2D (after_turn.ball_pos.getX (), after_turn.ball_pos.getY (), "" << (i),
                    "5555ff"));
            LOG_POL (0, << _2D << L2D (after_turn.ball_pos.getX (), after_turn.ball_pos.getY (),
                    final.ball_pos.getX (), final.ball_pos.getY (), "000000"));
            continue;
        }


        // printf ("logging\n");
//        double dist = after_turn.my_pos.distance (after_turn.ball_pos);
//        double prob = WSinfo::get_tackle_probability (false, intermediate.ball_pos);
        double prob = Tools::get_tackle_success_probability (after_turn.my_pos, after_turn.ball_pos,
                                                             after_turn.my_angle.get_value ());
        double speed = final.ball_vel.norm ();

        LOG_POL (0, << _2D << L2D (initial.ball_pos.getX (), initial.ball_pos.getY (),
                                   after_turn.ball_pos.getX (), after_turn.ball_pos.getY (), "5555ff"));
        LOG_POL (0, << _2D << STRING2D (after_turn.ball_pos.getX (), after_turn.ball_pos.getY (), "" << (i),
                "5555ff"));
        LOG_POL (0, << _2D << L2D (after_turn.ball_pos.getX (), after_turn.ball_pos.getY (),
                                   final.ball_pos.getX (), final.ball_pos.getY (), "ff5555"));

        if (prob >= min_success && speed >= min_ball_speed)
        {
            if (/* prob >= min_success && speed >= min_ball_speed && */ abs (i) > max_angle)
            {
                max_angle  = abs (i);
                kick_angle = angle;
                kick_power = power;
                state      = KICK;
            }
            LOG_POL (0, << _2D << STRING2D (final.ball_pos.getX (), final.ball_pos.getY (), "" << (prob) << "/" << speed,
                    "ff5555"));
        }
        else
        {
            LOG_POL (0, << _2D << STRING2D (final.ball_pos.getX (), final.ball_pos.getY (), "" << (prob) << "/" << speed,
                    "000000"));
        }
        /*
        printf ("dist=%4.2f  final_lot=%4.2f  opt_t_angle=%-3f=> prob=%5.3f  speed=%4.2f\n",
                dist,
//                Tools::get_dist2_line (intermediate.my_pos, intermediate.my_pos + Vector (intermediate.my_angle), intermediate.ball_pos),
                min_target_lot,
                optimal_tackle_angle, // * final_tackle_angle,
                prob,
                speed);
         */
//        printf ("dist=%4.2f\t prob=%5.3f\t speed=%4.2f\t dir=%4.0f\n", dist, prob, speed, after.ball_vel.arg () / PI * 180);
    }

//    printf ("final max angle: %d\n", max_angle);
    return max_angle != -1;
}


bool KickNTackle::simulate_turn_tackle (double min_success, double min_ball_speed)
{
    Cmd cmd = Cmd ();
    MyState initial    = MyState ();
    MyState next       = MyState ();
    MyState after_turn = MyState ();
    MyState final      = MyState ();
    initial.get_from_WS ();

    // just simulate one more step to see where ball will be afterwards
    Tools::get_successor_state (initial, cmd.cmd_body, next);


    // === sim turn ============================================================================================
    // turning to angle as 2nd step
    double turn_diff = (Vector (next.ball_pos - next.my_pos).ARG () - initial.my_angle).get_value_mPI_pPI ();// / PI * 180.;
    basic->set_turn_inertia (turn_diff, initial.my_vel);
    basic->get_cmd (cmd);
    Tools::get_successor_state (initial, cmd.cmd_body, after_turn);
#if LOGGING && BASIC_LOGGING
    const Vector &dir_pos = after_turn.my_pos + (after_turn.my_angle) + (after_turn.my_angle) + (after_turn.my_angle) + (after_turn.my_angle);
#endif
    LOG_POL (0, << _2D << L2D (after_turn.ball_pos.getX (), after_turn.ball_pos.getY (),
            target.getX (), target.getY (), "ffff55"));
    LOG_POL (0, << _2D << L2D (after_turn.my_pos.getX (), after_turn.my_pos.getY (),
            dir_pos.getX (), dir_pos.getY (), "ffffff"));


    // === sim tackle ==========================================================================================
    bool success = find_optimal_tackle (after_turn, final, tackle_angle);
    if (! success)
    {
        printf ("unusable tackle parameter (2nd step)!!!!!\n");
        LOG_POL (0, << _2D << L2D (initial.ball_pos.getX (), initial.ball_pos.getY (),
                after_turn.ball_pos.getX (), after_turn.ball_pos.getY (), "5555ff"));
        LOG_POL (0, << _2D << L2D (after_turn.ball_pos.getX (), after_turn.ball_pos.getY (),
                final.ball_pos.getX (), final.ball_pos.getY (), "000000"));
//        return false;
    }
    else
    {
        LOG_POL (0, << _2D << L2D (initial.ball_pos.getX (), initial.ball_pos.getY (),
                after_turn.ball_pos.getX (), after_turn.ball_pos.getY (), "5555ff"));
        LOG_POL (0, << _2D << L2D (after_turn.ball_pos.getX (), after_turn.ball_pos.getY (),
                final.ball_pos.getX (), final.ball_pos.getY (), "ff5555"));
    }


    // printf ("logging\n");
//        double dist = after_turn.my_pos.distance (after_turn.ball_pos);
//        double prob = WSinfo::get_tackle_probability (false, intermediate.ball_pos);
    double prob = Tools::get_tackle_success_probability (after_turn.my_pos, after_turn.ball_pos,
                                                         after_turn.my_angle.get_value ());
    double speed = final.ball_vel.norm ();


    if (prob >= min_success && speed >= min_ball_speed) {
        state      = TURN;
        turn_angle = turn_diff;
        last_simulation_time = WSinfo::ws->time;
        LOG_POL (0, << _2D << STRING2D (final.ball_pos.getX (), final.ball_pos.getY (), "" << (prob) << "/" << speed,
                "ff5555"));
        return true;
    }

    LOG_POL (0, << _2D << STRING2D (final.ball_pos.getX (), final.ball_pos.getY (), "" << (prob) << "/" << speed,
            "000000"));
    return false;
}


bool KickNTackle::simulate_tackle (double min_success, double min_ball_speed)
{
    Cmd cmd = Cmd ();
    MyState initial    = MyState ();
    MyState final      = MyState ();
    initial.get_from_WS ();

    // === sim tackle ==========================================================================================
    bool success = find_optimal_tackle (initial, final, tackle_angle);
    LOG_POL (0, << _2D << L2D (initial.ball_pos.getX (), initial.ball_pos.getY (),
            target.getX (), target.getY (), "ffff55"));
    if (! success)
    {
        printf ("unusable tackle parameter (3rd step)!!!!!\n");
        LOG_POL (0, << _2D << L2D (initial.ball_pos.getX (), initial.ball_pos.getY (),
                final.ball_pos.getX (), final.ball_pos.getY (), "000000"));
//        return false;
    }
    else
    {
        LOG_POL (0, << _2D << L2D (initial.ball_pos.getX (), initial.ball_pos.getY (),
                final.ball_pos.getX (), final.ball_pos.getY (), "ff5555"));
    }

    // printf ("logging\n");
//        double dist = after_turn.my_pos.distance (after_turn.ball_pos);
//        double prob = WSinfo::get_tackle_probability (false, intermediate.ball_pos);
    double prob = Tools::get_tackle_success_probability (initial.my_pos, initial.ball_pos,
                                                         initial.my_angle.get_value ());
    double speed = final.ball_vel.norm ();


    if (prob >= min_success && speed >= min_ball_speed) {
        state        = TACKLE;
        // printf ("tackle_angle ====== %f\n", tackle_angle);
        last_simulation_time = WSinfo::ws->time;
        LOG_POL (0, << _2D << STRING2D (final.ball_pos.getX (), final.ball_pos.getY (), "" << (prob) << "/" << speed,
                "ff5555"));
        return true;
    }

    LOG_POL (0, << _2D << STRING2D (final.ball_pos.getX (), final.ball_pos.getY (), "" << (prob) << "/" << speed,
            "000055"));
    return false;
}


bool KickNTackle::find_optimal_kick (const MyState &initial, MyState &next, const Vector &target_pos, int steps,
                                     double &angle, double &velocity)
{
    //double ball_dist = initial.ball_pos.distance (target_pos);
    velocity = Tools::get_ballspeed_for_dist_and_steps (initial.ball_pos.distance (target_pos), steps);
    angle = (target_pos - initial.my_pos).arg ();

    Cmd cmd = Cmd ();

    kick->set_state (initial.my_pos, initial.my_vel, initial.my_angle, initial.ball_pos, initial.ball_vel,
                     Vector (-55, -55), 0, -1, 0);
    kick->kick_in_dir_with_initial_vel (velocity, angle);
    if (! kick->get_cmd (cmd))
    {
        return false; // kick not possible with given parameters
    }
    Tools::get_successor_state (initial, cmd.cmd_body, next);

    return true;
}


bool KickNTackle::find_optimal_tackle (const MyState &initial, MyState &next, double &middle_angle)
{
    MyState after;
    Cmd cmd = Cmd ();
    int iterations = 0;

//    LOG_POL (0, << _2D << C2D (initial.my_pos.getX (), initial.my_pos.getY (),
//            .1, "000000"));
//    LOG_POL (0, << _2D << C2D (initial.my_pos.getX (), initial.my_pos.getY (),
//            .3, "000000"));
    ANGLE me_to_ball     = ANGLE (initial.ball_pos.getX () - initial.my_pos.getX (),
                                  initial.ball_pos.getY () - initial.my_pos.getY ());
    ANGLE ball_to_target = ANGLE (target.getX () - initial.ball_pos.getX (),
                                  target.getY () - initial.ball_pos.getY ());
    LOG_POL (0, << _2D << L2D (initial.ball_pos.getX (), initial.ball_pos.getY (),
            target.getX (), target.getY (), "00ff00"));
    double lower_angle, higher_angle;
    double lower_lot, middle_lot, higher_lot;
    double min_lot, max_lot;

//    middle_angle = ANGLE (initial.ball_vel.arg () - 1.5 * ball_to_target.get_value ()).get_value_mPI_pPI ();
//    middle_angle = ANGLE (ball_to_target.get_value () - 1.85 * initial.ball_vel.arg ()).get_value_mPI_pPI ();
//    middle_angle = (initial.my_angle - ball_to_target).get_value_mPI_pPI ();
    middle_angle = (ball_to_target - me_to_ball).get_value_mPI_pPI ();
    // visualize start angle
//    LOG_POL (0, << _2D << L2D (initial.ball_pos.getX (), initial.ball_pos.getY (),
//             (initial.ball_pos + ANGLE (middle_angle)).getX (), (initial.ball_pos + ANGLE (middle_angle)).getY (), "ffff00"));
    lower_angle = middle_angle - .35; // 20°
    if (lower_angle < -PI) { lower_angle = -PI; }
    higher_angle = middle_angle + .35;
    if (lower_angle >  PI) { lower_angle =  PI; }

    lower_lot  = calculate_anglediff_after_tackle (initial, after,  lower_angle, ball_to_target);
    middle_lot = calculate_anglediff_after_tackle (initial, after, middle_angle, ball_to_target);
    higher_lot = calculate_anglediff_after_tackle (initial, after, higher_angle, ball_to_target);

    do
    {
        if (iterations++ > 6)
        {
            bool result = middle_lot > 0.1;
            if (! result)
            {
                printf ("no minimum tackle parameter found!!!!\n");
            }
            return result;
        }
//        printf (" tackle angl   (%.4f, %.4f, %.4f)\n", lower_angle, middle_angle, higher_angle);
//        printf (" tackle angle diffs   (%.4f, %.4f, %.4f)\n", lower_lot, middle_lot, higher_lot);
        min_lot = min (lower_lot, min (middle_lot, higher_lot));
        max_lot = max (lower_lot, max (middle_lot, higher_lot));

        if (min_lot == lower_lot)
        {
            if (max_lot == higher_lot)
            {
                higher_angle = middle_angle;
                higher_lot   = middle_lot;
                middle_angle = lower_angle;
                middle_lot   = lower_lot;
                lower_angle  = 2 * middle_angle - higher_angle; // interpolate
                lower_lot    = calculate_anglediff_after_tackle (initial, after, lower_angle, ball_to_target);
            }
        }
        else if (min_lot == middle_lot)
        {
            if (max_lot == higher_lot)
            {
                // lower_angle  = lower_angle;
                higher_angle = middle_angle;
                higher_lot   = middle_lot;
            }
            else
            {
                lower_angle = middle_angle;
                lower_lot   = middle_lot;
                // higher_power = higher_power;
            }
            if (lower_angle > higher_angle)
            {
                higher_angle += (2 * PI);
            }
            middle_angle = (higher_angle + lower_angle) / 2;
            if (middle_angle > (PI))
            {
                middle_angle -= (2 * PI);
            }
            middle_lot = calculate_anglediff_after_tackle (initial, after, middle_angle, ball_to_target);
        }
        else // (min_lot == higher_lot)
        {
            if (max_lot == lower_lot)
            {
                lower_angle  = middle_angle;
                lower_lot    = middle_lot;
                middle_angle = higher_angle;
                middle_lot   = higher_lot;
                higher_angle = 2 * middle_angle - lower_angle; // interpolate
                higher_lot = calculate_anglediff_after_tackle (initial, after, higher_angle, ball_to_target);
            }
        }
    }
    while (middle_lot > 0.04); // 2.5°

//    printf ("optimal tackle angle %f\n", middle_angle);
//    printf ("greedy tackle iterations %d\n", iterations);

    calculate_anglediff_after_tackle (initial, next, middle_angle, ball_to_target);
    return true;
}


double KickNTackle::calculate_anglediff_after_tackle (const MyState &initial, MyState &after, const double tackle_angle, const ANGLE ball_2_target)
{
    Cmd cmd = Cmd ();

    basic->set_tackle (tackle_angle * 180 / PI, false);
    basic->get_cmd (cmd);
    Tools::get_successor_state (initial, cmd.cmd_body, after);

    return after.ball_vel.ARG ().diff (ball_2_target);
}


bool KickNTackle::get_cmd (Cmd &cmd)
{
//    printf ("************* get cmd *********************************************************\n");
//    printf ("kickntackle get_cmd (%d, %d, %d)\n", last_simulation_time, last_execution_time, WSinfo::ws->time);
    bool cmd_set = false;
    if (last_simulation_time != WSinfo::ws->time)
    {
        // no valid data to be executed
        return cmd_set;
    }
//    printf ("kickntackle get_cmd (%d)\n", state);

    if (state == KICK)
    {
//        printf ("execute kick\n");
        kick->kick_in_dir_with_initial_vel (kick_power, kick_angle);
        cmd_set = kick->get_cmd (cmd);
    }
    else if (state == TURN)
    {
//        printf ("execute turn (by %f)\n", turn_angle);
        basic->set_turn_inertia (turn_angle);
        cmd_set = basic->get_cmd (cmd);
    }
    else if (state == TACKLE)
    {
//        printf ("execute tackle (with %f)\n", tackle_angle);
        basic->set_tackle (-tackle_angle * 180 / PI, false);
        cmd_set = basic->get_cmd (cmd);
    }

    if (cmd_set)
    {
        last_execution_time = WSinfo::ws->time;
    }

    return cmd_set;
}
