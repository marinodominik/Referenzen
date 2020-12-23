#ifndef INTERCEPT_BALL_BMS_H_
#define INTERCEPT_BALL_BMS_H_

/* This behavior implements a hand-coded intercept.
   The original code was written by Alexander Sung, and imported
   into our old move layout by Martin Riedmiller, who also
   added some improvements. It has finally been converted to
   our new concept of behaviors by Manuel Nickschas in 2003.

   Just call get_cmd(Cmd&); no further function calls are necessary.
   Return value will always be true.
*/


#include "../../basics/Cmd.h"
#include "../base_bm.h"
#include "log_macros.h"

#include "tools.h"
#include "options.h"
#include "ws_info.h"
#include "log_macros.h"

class InterceptBall : public BodyBehavior {
  static bool initialized;

  bool final_state();
  static Angle calculate_deviation_threshold(double distance);
  static bool is_destination_reachable(const Vector& destination, Vector my_pos, Vector my_vel, 
				       ANGLE my_ang, int turns);
  static bool is_destination_reachable2(const Vector& destination, Vector my_pos, Vector my_vel, 
				       ANGLE my_ang, int maxcycles, bool riskyIntercept=false);
  
  ANGLE my_angle_to(const Vector & my_pos, const ANGLE &my_angle, 
				   Vector target);
 public:
  InterceptBall();
  virtual ~InterceptBall();
  bool get_cmd(Cmd & cmd, const Vector & my_pos,const Vector & my_vel, 
	       const ANGLE my_ang, 
	       Vector ball_pos, Vector ball_vel, int &num_cycles, bool riskyIntercept=false);
  bool get_cmd(Cmd & cmd, const Vector & my_pos,const Vector & my_vel, 
	       const ANGLE my_ang, 
	       Vector ball_pos, Vector ball_vel);
  bool get_cmd(Cmd &cmd);
  bool get_cmd(Cmd &cmd, int &num_cycles);
  
  static bool init(char const * conf_file, int argc, char const* const* argv);

  /////////////////////// TG ///////////////////////
  private:
    Angle ivRequestedTurnAngle;
    long  ivValidityOfRequestedTurnAngle;
    static double cvBallPlayerDistanceAtInterception;
  public:
    bool  get_cmd_arbitraryPlayer( PPlayer player,
                                        Cmd & cmd, 
                                        const Vector & my_pos,
                                        const Vector & my_vel, 
                                        const ANGLE my_ang, 
                                        Vector ball_pos, 
                                        Vector ball_vel, 
                                        int &num_cycles,
                                        int maxcycles = 30,
                                        bool riskyIntercept = false);
    bool is_destination_reachable2_arbitraryPlayer
                               (PPlayer player,
                                const Vector& destination, 
                                Vector my_pos,
                                Vector my_vel, 
                                ANGLE my_ang, 
                                int maxcycles, 
                                bool riskyIntercept=false);
    bool checkForOptimzed1StepIntercept( PPlayer player,
                                         Cmd & cmd,
                                         Vector ballDestination,
                                         Vector myPos,
                                         Vector myVel,
                                         ANGLE  myAng );
    void setRequestForTurnAngle(Angle turnAngle, long timeOfValidity);
    bool get_cmd(Cmd &cmd, int &num_cycles, double &resBallDist);//TG09
    static double getBallPlayerInterceptDistanceForRecentRequest();
};

#endif
