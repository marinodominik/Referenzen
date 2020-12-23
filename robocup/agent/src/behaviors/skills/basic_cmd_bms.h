#ifndef _BASIC_CMD_BMS_H_
#define _BASIC_CMD_BMS_H_

/** This class contains the interface to the base commands of soccer server. */

#include "../base_bm.h"
#include "tools.h"

class BasicCmd : public BodyBehavior {
private:
    static bool initialized;

    enum bm_types {
        KICK,
        TURN,
        TURN_INERTIA,
        DASH,
        SIDE_DASH,
        TACKLE,
        CATCH,
        MOVE
    };

    int    type;
    long   type_set;

    double power;
    double moment;
    Vector vel;
    ANGLE  direction;
    Vector pos;
    bool   foul;
    int    priority;

    void do_turn_inertia (Cmd_Body &cmd, double moment, Vector const & vel);

public:
    BasicCmd ();
    virtual ~BasicCmd ();
    static bool init (char const * conf_file, int argc, char const* const* argv);


    void set_kick (double power, ANGLE direction);

    /**
     * Set turn as next command
     *
     * @param moment [in]   angle to turn [-Pi,Pi]
     */
    void set_turn (double moment);
    /**
     * Set turn as next command based on current player's inertia (speed, direction etc.)
     *
     * @param moment [in]   angle to turn [-Pi,Pi]
     */
    void set_turn_inertia (double moment);
    /**
     * Set turn as next command based on current player's inertia (speed, direction etc.),
     * but given player's velocity
     *
     * @param moment [in]   angle to turn [-Pi,Pi]
     * @param vel [in]      players velocity
     */
    void set_turn_inertia (double moment, Vector const & vel);

    void set_dash (double power, int priority = 0);
    void set_dash (double power, ANGLE direction);

    /**
     * Set tackle as next command
	 *
     * @param angle [in] define kick angle [-180,180] (prior versions of
	 *					 soccer server (<12) interpret this parameter as
	 * 					 power [-100,100]
     * @param foul [in]  foul or ball tackle
     */
    void set_tackle (double angle, bool foul = false);
    /**
     * Set tackle as next command
	 *
     * @param angle [in] define kick angle
     * @param foul [in]  foul or ball tackle
     */
    void set_tackle (ANGLE const &angle, bool foul = false);

    void set_catch (ANGLE direction);

    void set_move (Vector pos);

    /**
     * get actual command
     * @param cmd [in,out]  adjust command according to previously set_* methods
     * @return              true, if get_cmd was successfully
     */
    bool get_cmd (Cmd &cmd);
};

#endif