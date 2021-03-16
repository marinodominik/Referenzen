//
// Created by Eicke Godehardt on 2/24/17.
//

#ifndef KICK_N_TACKLE_BMS_H
#define KICK_N_TACKLE_BMS_H

#include "intercept_ball_bms.h"
// #include "skills/neuro_kick_wrapper_bms.h"
#include "basic_cmd_bms.h"
#include "one_step_kick_bms.h"
#include "basics/mystate.h"


class KickNTackle : public BodyBehavior {
    static bool initialized;

    enum tackle_state {KICK, TURN, TACKLE};
    tackle_state state;

    int last_simulation_time;
    int last_execution_time;

    ANGLE  kick_angle;
    double kick_power;
//    double kick_distance;
    double turn_angle;
    double tackle_angle;
    double last_probability;
    Vector target;

    bool simulate_all        (double min_success, double min_ball_speed);
    bool simulate_turn_tackle(double min_success, double min_ball_speed);
    bool simulate_tackle     (double min_success, double min_ball_speed);

    /**
     * Finds optimal kick parameters (angle and power) based on a greedy approach.
     * Only if possible with less than 10 iterations!
     *
     * @param initial    [in]  initial state
     * @param next      [out]  state after kick
     * @param target_pos [in]
     * @param steps      [in]
     * @param angle     [out]
     * @param power     [out]
     * @return                 returns success indicator
     */
    bool find_optimal_kick   (const MyState &initial, MyState &next, const Vector &target_pos, int steps, double &angle, double &power);
    bool find_optimal_tackle (const MyState &initial, MyState &next, double &tackle_angle);
    /**
     * Helper method
     * @param initial
     * @param next
     * @param target_pos
     * @param angle
     * @param power
     * @param steps
     * @return
     */
    double calculate_anglediff_after_tackle (const MyState &initial, MyState &next, const double angle, const ANGLE ball_to_target);

    InterceptBall    *interceptball;
//    NeuroKickWrapper *neuroKick;
    BasicCmd         *basic;
    OneStepKick      *kick;
public:
    KickNTackle ();
    virtual ~KickNTackle ();

    /**
     *
     * @param success       [0,1] minimal success probability
     * @param target        target point
     * @param min_ball_speed
     * @return
     */
    bool is_possible (const Vector &target, double success = .8, double min_ball_speed = 1.5);
    bool get_cmd (Cmd & cmd);

    static bool init (char const * conf_file, int argc, char const* const* argv);
};

#endif // KICK_N_TACKLE_BMS_H
