#include "tools.h"

#include <sys/time.h>

#include "options.h"
#include "log_macros.h"
#include "blackboard.h"
#include "policy/policy_tools.h"
#include "policy/planning.h"
#include "behaviors/skills/basic_cmd_bms.h"
#include "serverparam.h"


#if 1
#define DBLOG_POL(LLL,XXX) LOG_POL(LLL,XXX)
#define DBLOG_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D<<XXX)
#else
#define DBLOG_DRAW(LLL,XXX)
#define DBLOG_POL(LLL,XXX) 
#endif

#define MIN(X,Y) ((X<Y)?X:Y)

int Tools::num_powers= -1;
double Tools::ball_decay_powers[MAX_NUM_POWERS];
AState Tools::state;
int Tools::state_valid_at;
PPlayer Tools::modelled_player = WSinfo::me;

bool Tools::deduce_action_from_states( Cmd_Body&     action,
                                       const PPlayer player,       double distToOtherPlayer,
                                       const Vector& oldPlayerPos, Vector oldPlayerVel, ANGLE oldPlayerAng,
                                       const Vector& newPlayerPos, Vector newPlayerVel, ANGLE newPlayerAng,
                                       const bool    ignoreBall,   double distToBall,
                                       const Vector& oldBallPos,   Vector oldBallVel,
                                       const Vector& newBallPos,   Vector newBallVel
                                     )
{
    bool isIdentified = false;
    action.reset();


    double tolerance = 1.1;
    double threshold = 0.001;
    bool   fullstate = ServerOptions::fullstate_l && ServerOptions::fullstate_r;

    double ballViewNoiseMultiplicator = 0.125;
    double playerViewNoiseMultiplicator = 0.125;

    Vector oldPlayerAccel = ( newPlayerVel / player->decay ) - oldPlayerVel;
    Vector oldBallAccel;
    Vector playerToBall;



    // BALL
    const int BALL_ERROR          = -1;
    const int BALL_NOT_INFLUENCED =  0;
    const int BALL_INFLUENCED     =  1;

    int ballState = BALL_ERROR;

    if( !ignoreBall )
    {
        if( oldBallVel.norm() < threshold ) { oldBallVel = Vector(); }
        if( newBallVel.norm() < threshold ) { newBallVel = Vector(); }
        double ballViewNoise = 0;
        if( !fullstate ) { ballViewNoise = Tools::getViewNoiseFromDist( distToBall ) * ballViewNoiseMultiplicator; }

        oldBallAccel = ( newBallVel / ServerOptions::ball_decay ) - oldBallVel;
        playerToBall = oldPlayerPos - oldBallPos;

        bool possibleBallPos   = oldBallPos.distance( newBallPos ) - ballViewNoise <= ServerOptions::ball_speed_max * tolerance;
        bool possibleBallVel   =
                oldBallVel.norm() <= ServerOptions::ball_speed_max * tolerance &&
                newBallVel.norm() <= ServerOptions::ball_speed_max * tolerance;
        bool possibleBallAccel = oldBallAccel.norm() <= ServerOptions::ball_accel_max;


        if( possibleBallPos && possibleBallVel && possibleBallAccel )
        {
            bool expectedBallPos = oldBallPos.distance( newBallPos ) - ballViewNoise <= ( oldBallVel * ( 1 + ( ServerOptions::ball_rand * tolerance ) ) ).norm();
            bool expectedBallVel = newBallVel.isBetween(
                    ( oldBallVel * ServerOptions::ball_decay * ( 1 - ( ServerOptions::ball_rand * tolerance ) ) ),
                    ( oldBallVel * ServerOptions::ball_decay * ( 1 + ( ServerOptions::ball_rand * tolerance ) ) ) );

            if( expectedBallPos && expectedBallVel )
            {
                ballState = BALL_NOT_INFLUENCED;
            }
            else{
                ballState = BALL_INFLUENCED;
            }
        }
    }

    // PLAYER
    if( oldPlayerVel.norm() < threshold ) { oldPlayerVel = Vector(); }
    if( newPlayerVel.norm() < threshold ) { newPlayerVel = Vector(); }
    double playerViewNoise = 0;
    if( !fullstate ) { playerViewNoise = Tools::getViewNoiseFromDist( distToOtherPlayer ) * playerViewNoiseMultiplicator; }

    bool possiblePlayerPos   = oldPlayerPos.distance( newPlayerPos ) - playerViewNoise <= player->speed_max * tolerance;
    bool possiblePlayerVel   =
            oldPlayerVel.norm() <= player->speed_max * tolerance &&
            newPlayerVel.norm() <= player->speed_max * tolerance;
//    bool possiblePlayerAccel = oldPlayerAccel.norm() <= 1; // TODO: 1 durch ServerOptions::player_accel_max ersetzen (existiert noch nicht)
    bool possiblePlayerAng   = false;

    ANGLE diffAng = oldPlayerAng.diff( newPlayerAng );
    ANGLE maxDiffAng(degreeInRadian( ( 180 * tolerance ) / ( 1.0 + player->inertia_moment * oldPlayerVel.norm() ) )); // TODO: 180 durch ServerOptions::maxmoment ersetzen (existiert noch nicht)

    if( diffAng <= maxDiffAng)
    {
        possiblePlayerAng = true;
    }


    if( possiblePlayerPos && possiblePlayerVel && possiblePlayerAng )
    {
        double expectedPlayerMaxPosDiff = ( oldPlayerVel * ( 1 + ( ServerOptions::player_rand * tolerance ) ) ).norm();
        if( expectedPlayerMaxPosDiff < threshold ) { expectedPlayerMaxPosDiff = threshold; }

        bool expectedPlayerPos = oldPlayerPos.distance( newPlayerPos ) - playerViewNoise <= expectedPlayerMaxPosDiff;
        bool expectedPlayerVel = newPlayerVel.isBetween(
                ( oldPlayerVel * player->decay * ( 1 + ServerOptions::player_rand ) * tolerance ),
                ( oldPlayerVel * player->decay * ( 1 - ServerOptions::player_rand ) * ( 1 - ( tolerance - 1 ) ) ) )
                || ( oldPlayerVel.sqr_norm() < threshold && newPlayerVel.sqr_norm() < threshold );
        bool expectedPlayerAng = oldPlayerAng.diff( newPlayerAng ) < 0.02;

//        cout    << endl
////                << "      ||| oldNewDist = " << oldPlayerPos.distance( newPlayerPos ) << endl
////                << "      ||| playerViewNoise = " << playerViewNoise << endl
////                << "      ||| expectedPlayerMaxPosDiff = " << expectedPlayerMaxPosDiff << endl
//                << "expPos = " << expectedPlayerPos << "   expVel = " << expectedPlayerVel << "   expAng = " << expectedPlayerAng << endl
////                << "Pos:    old=" << oldPlayerPos << " new=" << newPlayerPos << endl
////                << "Vel:    old=" << oldPlayerVel << " new=" << newPlayerVel << endl
////                << "Vel-length: old=" << oldPlayerVel.norm() << "                new=" << newPlayerVel.norm() << endl
////                << "maxVel: old=" << oldPlayerVel * player->decay * ( 1 + ServerOptions::player_rand ) * tolerance << endl
////                << "minVel: old=" << oldPlayerVel * player->decay * ( 1 - ServerOptions::player_rand ) * ( 1 - ( tolerance - 1 ) ) << endl
//                << "Ang:    old=" << oldPlayerAng << "                new=" << newPlayerAng << endl
//                << flush;

        if( ballState != BALL_ERROR ) // BALL RELATED ACTIONS
        {

            if( expectedPlayerPos && expectedPlayerAng && ballState == BALL_INFLUENCED && // KICK
                    oldPlayerPos.distance(oldBallPos) <= player->kick_radius * tolerance // - ball and player radius?
            )
            {
                // KICK
                double kickPower = 0;
                ANGLE  kickDirection = oldBallAccel.ARG() - oldPlayerAng;

                double dir_diff  = fabs( ( playerToBall.ARG() - oldPlayerAng ).get_value_mPI_pPI() );
                double dist_ball = playerToBall.norm() - ServerOptions::player_size - ServerOptions::ball_size; /* TODO: von heterogenem Typ abhängig machen ( ServerOptions::player_size durch player->size ersetzen (existiert noch nicht)) */


                kickPower = (oldBallAccel.norm() / player->kick_power_rate)
                        / ( 1.0 - 0.25*dir_diff/M_PI - 0.25*dist_ball/ player->kick_radius );

                action.set_kick( kickPower, kickDirection );
                isIdentified = true;
            }
            else
            {

                Vector playerToBallAbs = Vector(playerToBall).ROTATE( -oldPlayerAng );
                double tackleDist = playerToBallAbs.getX() > 0.0 ? ServerOptions::tackle_dist : ServerOptions::tackle_back_dist;

                /* ServerOptions::foul_exponent kann nicht berücksichtigt werden,
                 * weil dazu ein Foul notwendig wäre, was nur möglich ist,
                 * wenn ein anderer Spieler den Ball in seinem kick_radius hat.
                 * Da andere Spieler bei der Auswertung nicht berücksichtigt werden,
                 * kann kein Foul festgestellt werden. */
                double tackleProb = ( pow( fabs( playerToBallAbs.getX() ) / tackleDist,                  ServerOptions::tackle_exponent )
                                    + pow( fabs( playerToBallAbs.getY() ) / ServerOptions::tackle_width, ServerOptions::tackle_exponent ) );

                if( expectedPlayerPos && expectedPlayerAng && ballState == BALL_INFLUENCED && tackleProb < 1.0 * tolerance ) // TACKLE
                {
                    // TACKLE
                    ANGLE tackleAngle = oldBallAccel.ARG();

                    action.set_tackle(tackleAngle, false);
                    isIdentified = true;
                }
//                else
//                {
//                    double longestCatchDist = sqrt( ( ServerOptions::catchable_area_l * ServerOptions::catchable_area_l ) + ( ( ServerOptions::catchable_area_w / 2 ) * ( ServerOptions::catchable_area_w / 2 ) ) );
//
//                    if( player == WSinfo::get_opponent_by_number( WSinfo::ws->his_goalie_number )
//                        && RIGHT_PENALTY_AREA.inside( oldPlayerPos )
//                        && oldPlayerPos.distance(oldBallPos) <= longestCatchDist * tolerance
//                        && false // deaktiviert, weil:
//                    ) // nötige aber nicht sicherstellende Kreterien
//                    {
//                        // CATCH
//
//                        ANGLE catchAngle; // NOT DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//
//                        action.set_catch(catchAngle);
//                        isIdentified = true;
//                    }
//                }
            }
        } // END if( ballState != BALL_ERROR )
        if( !isIdentified && !expectedPlayerAng ) // TURN
        {
            // TURN
            ANGLE turnAngle = ANGLE( ANGLE( newPlayerAng.get_value_0_p2PI() - oldPlayerAng.get_value_0_p2PI() ).get_value_mPI_pPI() * ( 1.0 + player->inertia_moment * oldPlayerVel.norm() ) );

            action.set_turn( turnAngle );
            isIdentified = true;
        }
        else if( !isIdentified && !expectedPlayerPos && !expectedPlayerVel && expectedPlayerAng ) // DASH
        {

            // DASH
            double dashPower = 0;
            ANGLE  dashDirection;

            double directionInDegree = ( oldPlayerAng - oldPlayerAccel.ARG() ).get_value_mPI_pPI() * (180.0 / PI);

            if ( ServerOptions::dash_angle_step > 1.0e-10 )
            {
                // The dash direction is discretized by server::dash_angle_step
            	directionInDegree = ServerOptions::dash_angle_step * rint( directionInDegree / ServerOptions::dash_angle_step );
            }

            dashDirection = ANGLE(directionInDegree / (180.0 / PI));

            double dir_rate = ( fabs(directionInDegree) > 90.0
                                ? ServerOptions::back_dash_rate - ((ServerOptions::back_dash_rate - ServerOptions::side_dash_rate) * (1.0 - (fabs(directionInDegree) - 90.0) / 90.0))
                                : ServerOptions::side_dash_rate + ((1.0                           - ServerOptions::side_dash_rate) * (1.0 -  fabs(directionInDegree)         / 90.0)) );

            LOG_POL(0, << _2D << VL2D(oldPlayerPos, oldPlayerPos + oldPlayerVel + oldPlayerAccel, "#0000FF"));

            dashPower = (oldPlayerAccel.norm() / dir_rate ) / player->dash_power_rate;

            action.set_dash( dashPower, dashDirection );
            isIdentified = true;
        }

        if( !isIdentified )
        {
            action.set_none();
            isIdentified = true;
        }

    } // END if( possiblePlayerPos && possiblePlayerVel && possiblePlayerAng )
//    else
//    {
//
//        cout    << endl
//                << "possPos = " << possiblePlayerPos << "   possVel = " << possiblePlayerVel << "   possAng = " << possiblePlayerAng << endl
//                << "Pos:        old = " << oldPlayerPos << "   new = " << newPlayerPos << endl
//                << "Vel:        old = " << oldPlayerVel << "   new = " << newPlayerVel << endl
//                << "Ang:        old = " << oldPlayerAng << "                  new = " << newPlayerAng << endl
//                << flush;
//        if( !possiblePlayerAng )
//        {
//            cout << "   !!!!! diffAng = " << diffAng << "   <=   maxDiffAng = " << maxDiffAng << endl << flush;
//        }
//    }

    return isIdentified;
}

double Tools::getViewNoiseFromDist( double distance )
{
	if(      distance <=  1.2 ) return 0.0;
	else if( distance <=  3.4 ) return 0.1;
	else if( distance <=  5.7 ) return 0.2;
	else if( distance <=  7.7 ) return 0.3;
	else if( distance <=  9.4 ) return 0.4;
	else if( distance <= 11.5 ) return 0.5;
	else if( distance <= 14.1 ) return 0.6;
	else if( distance <= 15.6 ) return 0.7;
	else if( distance <= 17.2 ) return 0.8;
	else if( distance <= 19.1 ) return 0.9;
	else if( distance <= 21.1 ) return 1.0;
	else if( distance <= 23.3 ) return 1.1;
	else if( distance <= 25.7 ) return 1.2;
	else if( distance <= 28.5 ) return 1.4;
	else if( distance <= 31.5 ) return 1.5;
	else if( distance <= 34.8 ) return 1.7;
	else if( distance <= 38.4 ) return 1.8;
	else if( distance <= 42.5 ) return 2.1;
	else if( distance <= 46.9 ) return 2.2;
	else if( distance <= 51.9 ) return 2.5;
	else if( distance <= 57.3 ) return 2.7;
	else if( distance <= 63.4 ) return 3.1;
	else return 999999;
}

std::ostream& operator<< (std::ostream& o, const MyState& s) 
{
  return o <<s.my_pos << s.my_vel << s.ball_pos<<s.ball_vel<<" "
	   <<RAD2DEG(s.my_angle.get_value())<<" "<<s.my_pos.distance(s.ball_pos);
}
Angle Tools::get_abs_angle(Angle angle){
  Angle result;
  result = get_angle_between_null_2PI(angle);
  if(result>PI)
    result -= 2*PI;
  return(fabs(result));
}

ANGLE Tools::my_angle_to(Vector target){
  ANGLE result;
  target -= WSinfo::me->pos;
  result = target.ARG() - WSinfo::me->ang;
  return result;
}

ANGLE Tools::my_abs_angle_to(Vector target) {
  ANGLE result;
  result = my_angle_to(target);
  return result + WSinfo::me->ang;
}

ANGLE Tools::my_neck_angle_to(Vector target) {
  ANGLE result;
  target -= WSinfo::me->pos;
  result = target.ARG() - WSinfo::me->neck_ang;
  return result;
}

/** returns expected relative body turn angle */
ANGLE Tools::my_expected_turn_angle() {
  if(!WSinfo::current_cmd->cmd_body.is_cmd_set()) return ANGLE(0);
  if(WSinfo::current_cmd->cmd_body.get_type() != Cmd_Body::TYPE_TURN) return ANGLE(0);
  double turnangle=0;
  WSinfo::current_cmd->cmd_body.get_turn(turnangle);
  //cout << "turnangle: " << turnangle;
  if(turnangle>PI) turnangle-=2*PI; 
  double newangle=turnangle/(1.0+WSinfo::me->inertia_moment*WSinfo::me->vel.norm());
  return ANGLE(newangle);
}

double Tools::moment_to_turn_neck_to_abs(ANGLE abs_dir_ang) {

  double abs_dir = abs_dir_ang.get_value();
  Angle minang=(my_minimum_abs_angle_to_see()+ANGLE(.5*next_view_angle_width().get_value())).get_value();
  Angle maxang=(my_maximum_abs_angle_to_see()-ANGLE(.5*next_view_angle_width().get_value())).get_value();
  if(minang<maxang) {
    if(abs_dir<minang) abs_dir=minang;
    if(abs_dir>maxang) abs_dir=maxang;
  } else {
    if(abs_dir>maxang && abs_dir<(minang-maxang)/2.+maxang) abs_dir=maxang;
    else if(abs_dir<minang && abs_dir>maxang) abs_dir=minang;
  }      
  return Tools::get_angle_between_null_2PI(abs_dir-WSinfo::me->neck_ang.get_value()
					   -my_expected_turn_angle().get_value());
}

/** gets the maximum angle I could see (abs, right) */
ANGLE Tools::my_maximum_abs_angle_to_see() {
  ANGLE max_angle=WSinfo::me->ang+my_expected_turn_angle();
  max_angle+=ServerOptions::maxneckang+ANGLE(.5*next_view_angle_width().get_value());
  return max_angle;
}

/** gets the minimum angle I could see (abs, left) */
ANGLE Tools::my_minimum_abs_angle_to_see() {
  ANGLE min_angle=WSinfo::me->ang+my_expected_turn_angle();
  min_angle+=ServerOptions::minneckang-ANGLE(.5*next_view_angle_width().get_value());
  return min_angle;
}

/** returns true if abs direction can be in view_angle with appropriate neck turn */
bool Tools::could_see_in_direction(ANGLE target_ang) {
  Angle minang=my_minimum_abs_angle_to_see().get_value();
  Angle maxang=my_maximum_abs_angle_to_see().get_value();
  Angle target=target_ang.get_value();

#if 0
  LOG_POL(0,"Tools: could see in direction: Minang :"<<RAD2DEG(minang)
	  <<" maxang: "<<RAD2DEG(maxang)<<" targetangle: "<<RAD2DEG(target))
#endif

  if(minang>maxang && (target<minang && target>maxang)) return false;
  if(minang<maxang && (target<minang || target>maxang)) return false;
  return true;
}


/* ridi 2006:  new computation */
/* ridi 2006: alternative computation: (not yet tested)
bool Tools::could_see_in_direction(ANGLE target_ang) {
  ANGLE my_next_angle=WSinfo::me->ang+my_expected_turn_angle();

  if(fabs(my_next_angle.diff(target_ang)) < ServerOptions::maxneckang+ANGLE(.5*next_view_angle_width().get_value()))
    return true;
  else 
    return false;
}
*/

bool Tools::could_see_in_direction(Angle target_ang) {
  return could_see_in_direction(ANGLE(target_ang));
}

double
Tools::get_foul_success_probability( Vector my_pos, 
                                     Vector ball_pos,
                                     double my_angle)
{
  bool foul = false;
  for (int i=0; i < WSinfo::valid_opponents.num; i++)
    if (WSinfo::is_ball_kickable_for( WSinfo::valid_opponents[i] ) )
      foul = true;
  if (foul)
    return get_tackle_success_probability( my_pos, 
                                           ball_pos, 
                                           my_angle,
                                           true );
  else
    return get_tackle_success_probability( my_pos, 
                                           ball_pos, 
                                           my_angle,
                                           false );
}

double Tools::get_tackle_success_probability( Vector my_pos,
                                              Vector ball_pos,
                                              double my_angle,
                                              bool   foul)
{
  double ret;
  Vector player_2_ball = ball_pos - my_pos;
  player_2_ball.rotate(-my_angle);
  double exponent = ServerOptions::tackle_exponent;
  if (foul) exponent = ServerOptions::foul_exponent;
  if (player_2_ball.getX() >= 0.0)
  {
    ret =   pow( player_2_ball.getX()/ServerOptions::tackle_dist,
                 exponent ) 
          + pow( fabs(player_2_ball.getY())/ServerOptions::tackle_width,
                 exponent );
  } 
  else 
  {
    double probPartX, probPartY;
    if ( fabs(ServerOptions::tackle_back_dist) < 0.001 )
      probPartX = 1.0;
    else
      probPartX = pow( player_2_ball.getX()/ServerOptions::tackle_back_dist,
                       exponent);
    if ( fabs(ServerOptions::tackle_width) < 0.001 )
      probPartY = 1.0;
    else
      probPartY = pow( fabs(player_2_ball.getY())/ServerOptions::tackle_width,
                       exponent);
    ret = probPartX + probPartY;
  }
#if 0
  LOG_POL(0,<<"Tools: tackle ret: "<<ret<<" tackle dist "<<ServerOptions::tackle_dist 
	  <<" tackle/foul exp "<<exponent 
	  <<" tackle width "<<ServerOptions::tackle_width 
	  <<" tackle back dist "<<ServerOptions::tackle_back_dist 
	  );

#endif

  if (ret >= 1.0) return 0.0;
  else if (ret < 0.0) return 1.0;
  else return (1.0 - ret);
}

Angle Tools::get_angle_between_mPI_pPI(Angle angle) {
  while (angle >= PI) angle -= 2*PI;
  while (angle < -PI) angle += 2*PI;
  return angle;
}

Angle Tools::get_angle_between_null_2PI(Angle angle) {
  while (angle >= 2*PI) angle -= 2*PI;
  while (angle < 0) angle += 2*PI;
  return angle;
}

double Tools::max(double a, double b) {
  if (a > b) return a;
  else return b;
}

double Tools::min(double a, double b) {
  if (a < b) return a;
  else return b;
}
int Tools::max(int a, int b) {
  if (a > b) return a;
  else return b;
}

int Tools::min(int a, int b) {
  if (a < b) return a;
  else return b;
}

int Tools::int_random(int n)
{
  static bool FirstTime = true;
  
  if ( FirstTime ){
    /* initialize the random number seed. */
    timeval tp;
    gettimeofday( &tp, NULL );
    srandom( (unsigned int) tp.tv_usec );
    FirstTime = false;
  }

  if ( n > 2 )
    return( random() % n );
  else if ( n == 2 )
    return( ( (random() % 112) >= 56 ) ? 0 : 1 );
  else if ( n == 1 )
    return(0);
  else
  {
    printf("int_random(%d) ?\n",n);
    printf( "You called int_random(<=0)\n" );
    return(0);
  }
}

double Tools::range_random(double lo, double hi)
{
  int x1 = int_random(10000);
  int x2 = int_random(10000);
  float r = (((float) x1) + 10000.0 * ((float) x2))/(10000.0 * 10000.0);
  return( lo + (hi - lo) * r );
}

int Tools::very_random_int(int n)
{
  int result = (int) range_random(0.0,(float)n);  /* rounds down */
  if ( result == n ) result = n-1;
  return(result);
}

Vector Tools::get_Lotfuss(Vector x1, Vector x2, Vector p){
  Vector r = x1 - x2;
  Vector a = x1 - p;
  double l = -(a.getX()*r.getX()+a.getY()*r.getY())/(r.getX()*r.getX()+r.getY()*r.getY());
  Vector ergebnis = x1 + l*r;
  return ergebnis;
}

double Tools::get_dist2_line(Vector x1, Vector x2, Vector p){
  //return get_Lotfuss(x1,x2,p).norm(); //this was wrong: corrected in 06/05
  return ( get_Lotfuss(x1,x2,p) - p ).norm();
}

long Tools::get_current_ms_time() { //returns time in ms since first call to this routine
  timeval tval;
  static long s_time_at_start= 0;
  if (gettimeofday(&tval,NULL))
    std::cerr << "\n something wrong with time mesurement";

  if ( 0 == s_time_at_start ) 
    s_time_at_start= tval.tv_sec;

  return (tval.tv_sec - s_time_at_start) * 1000 + tval.tv_usec / 1000;
}

void Tools::model_cmd_main(const Vector & old_my_pos, const Vector & old_my_vel, 
			   const ANGLE & old_my_ang,
			   const Vector & old_ball_pos, const Vector & old_ball_vel,
			   const Cmd_Body & cmd,
			   Vector & new_my_pos, Vector & new_my_vel,
			   ANGLE & new_my_ang,
			   Vector & new_ball_pos, Vector & new_ball_vel, const bool do_random) {
  const Angle a = old_my_ang.get_value_0_p2PI();
  Angle na;
  model_cmd_main(old_my_pos,old_my_vel,a,old_ball_pos,old_ball_vel,cmd,
		 new_my_pos,new_my_vel,na,new_ball_pos,new_ball_vel,do_random);
  new_my_ang = ANGLE(na);
}

void Tools::cmd2infostring(const Cmd_Body & cmd, char *info){
  double turn_angle = 0;
  double dash_power = 0;
  double kick_power = 0;
  double kick_angle = 0;
  switch(cmd.get_type()){
  case Cmd_Body::TYPE_DASH:
    cmd.get_dash(dash_power);
    sprintf(info,"Cmd_type DASH. power %g ",dash_power);
    break;
  case Cmd_Body::TYPE_KICK:
    cmd.get_kick(kick_power, kick_angle);
    sprintf(info,"Cmd_type KICK. power  %g  angle  %g",kick_power,(double)(RAD2DEG(kick_angle)));
  break;
  case Cmd_Body::TYPE_TURN:
    cmd.get_turn(turn_angle);
    sprintf(info,"Cmd_type TURN. angle %g",(double)(RAD2DEG(turn_angle)));
    break;
  }
}

double Tools::get_dash2stop(){
  Cmd_Body cmd_main;
  double   min_speed = WSinfo::me->vel.norm();
  Vector   dummypos;
  Vector   dummyvel;
  Angle    dummyang;

  double best_dash = 0;

  for(double dash = -100.;dash<100.; dash += 10.){
    cmd_main.unset_lock();
    cmd_main.set_dash(dash);
    model_player_movement(WSinfo::me->pos, WSinfo::me->vel, WSinfo::me->ang.get_value(), cmd_main,
			  dummypos, dummyvel, dummyang, false);

    double speed = fabs(dummyvel.norm());
    if(speed<min_speed){
      best_dash = dash;
      min_speed = speed;
    }
  }
  return best_dash;
}

bool Tools::get_cmd_to_stop(Cmd &cmd){
	bool cmd_set = false;
	BasicCmd basicCmd;

	ANGLE dash_angle = WSinfo::me->vel.ARG().get_value_0_p2PI()-DEG2RAD(180)-WSinfo::me->ang.get_value_0_p2PI();

//	LOG_POL(0,<<"Get_cmd_to_stop :maths:"<< stop.norm() << ":angle:" << RAD2DEG(dash_angle.get_value_0_p2PI()));

	double dash_effectiveness;
	if((dash_angle.get_value_0_p2PI() >= DEG2RAD(24.5) && dash_angle.get_value_0_p2PI() < DEG2RAD(67.5))
			|| (dash_angle.get_value_0_p2PI() >= DEG2RAD(292.5) && dash_angle.get_value_0_p2PI() < DEG2RAD(337.5)))
		dash_effectiveness = 0.7;
	else if((dash_angle.get_value_0_p2PI() >= DEG2RAD(67.5) && dash_angle.get_value_0_p2PI() < DEG2RAD(112.5))
			|| (dash_angle.get_value_0_p2PI() >= DEG2RAD(247.5) && dash_angle.get_value_0_p2PI() < DEG2RAD(292.5)))
		dash_effectiveness = 0.4;
	else if((dash_angle.get_value_0_p2PI() >= DEG2RAD(112.5) && dash_angle.get_value_0_p2PI() < DEG2RAD(157.5))
			|| (dash_angle.get_value_0_p2PI() >= DEG2RAD(202.5) && dash_angle.get_value_0_p2PI() < DEG2RAD(247.5)))
		dash_effectiveness = 0.5;
	else if(dash_angle.get_value_0_p2PI() >= DEG2RAD(157.5) && dash_angle.get_value_0_p2PI() < DEG2RAD(202.5))
		dash_effectiveness = 0.6;
	else
		dash_effectiveness = 1.0;

	double best_dash = ((Vector)-WSinfo::me->vel).norm() / WSinfo::me->effort / WSinfo::me->dash_power_rate / dash_effectiveness;

//	LOG_POL(0,<< "get_cmd_to_stop ::stop: "<< best_dash);

	basicCmd.set_dash(best_dash, dash_angle);
	cmd_set = basicCmd.get_cmd(cmd);

	return cmd_set;
}

void Tools::model_player_movement(const Vector & old_my_pos, const Vector & old_my_vel, 
				  const Angle & old_my_ang,
				  const Cmd_Body & cmd,
				  Vector & new_my_pos, Vector & new_my_vel,
				  Angle & new_my_ang,
				  const bool do_random) {

  Vector old_ball_pos, old_ball_vel, new_ball_pos, new_ball_vel;
  
  model_cmd_main(old_my_pos,old_my_vel,old_my_ang,old_ball_pos,old_ball_vel,cmd,
		 new_my_pos,new_my_vel,new_my_ang,new_ball_pos,new_ball_vel,do_random);

}


void Tools::model_cmd_main(const Vector & old_my_pos, const Vector & old_my_vel, 
			   const Angle & old_my_ang,
			   const Vector & old_ball_pos, const Vector & old_ball_vel,
			   const Cmd_Body & cmd,
			   Vector & new_my_pos, Vector & new_my_vel,
			   Angle & new_my_ang,
			   Vector & new_ball_pos, Vector & new_ball_vel, const bool do_random) {

  int old_stamina = 4000; // not used
  int new_stamina;
  model_cmd_main(old_my_pos,old_my_vel,old_my_ang,old_stamina,old_ball_pos,old_ball_vel,cmd,
		 new_my_pos,new_my_vel,new_my_ang,new_stamina, new_ball_pos,new_ball_vel,do_random);
}

void Tools::model_cmd_main(const Vector & old_my_pos, const Vector & old_my_vel, 
			   const Angle & old_my_ang, const int &old_my_stamina, 
			   const Vector & old_ball_pos, const Vector & old_ball_vel,
			   const Cmd_Body & cmd,
			   Vector & new_my_pos, Vector & new_my_vel,
			   Angle & new_my_ang, int &new_my_stamina,
			   Vector & new_ball_pos, Vector & new_ball_vel, const bool do_random) {

  double rand1= 0.0;
  double rand2= 0.0;
  Vector acc;
  double tmp;

#define GET_RANDOM(X,Y)((X)+drand48() *((Y)-(X)))

  double rmax_ball = ServerOptions::ball_rand * old_ball_vel.norm();
  double rmax_player = ServerOptions::player_rand * old_my_vel.norm();

  if(do_random){
    rand1=GET_RANDOM(-rmax_player,rmax_player);
    rand2=GET_RANDOM(-rmax_player,rmax_player);
  }

  
  /* computation of predicted state for ball and me */
  double turn_angle = 0;
  double dash_power = 0;
  Angle  dash_dir   = 0.0;
  double kick_power = 0;
  double kick_angle = 0;
  new_my_stamina = old_my_stamina; // default: do not change

  switch(cmd.get_type()){
  case Cmd_Body::TYPE_DASH:
    cmd.get_dash(dash_power, dash_dir);
    if(dash_power>0)
      new_my_stamina += (int)(-dash_power + modelled_player->stamina_inc_max);
    // assume that stamina increase is ok.
    else
      new_my_stamina += (int)(2*dash_power + modelled_player->stamina_inc_max);
    dash_power = dash_power*(1.0+rand1);
    break;
  case Cmd_Body::TYPE_KICK:
    cmd.get_kick(kick_power, kick_angle);
    kick_power = kick_power*(1.0+rand1);
    kick_angle = Tools::get_angle_between_mPI_pPI(kick_angle);
    kick_angle = kick_angle*(1.0+rand2);
    kick_angle = Tools::get_angle_between_null_2PI(kick_angle);
    break;
  case Cmd_Body::TYPE_TURN:
    cmd.get_turn(turn_angle);
    turn_angle = Tools::get_angle_between_mPI_pPI(turn_angle);
    turn_angle = turn_angle*(1.0+rand1);
    turn_angle = Tools::get_angle_between_mPI_pPI(turn_angle);
    break;
  case Cmd_Body::TYPE_TACKLE:
    double tackleAngle_input, foul;
    cmd.get_tackle (tackleAngle_input, foul);
    int tackleAngle = lround (tackleAngle_input);
    if (tackleAngle < 0)
    {
      tackleAngle += 360;
    }
    Tools::model_tackle_V12 (old_my_pos, ANGLE (old_my_ang), old_ball_pos, old_ball_vel, tackleAngle,
                             new_ball_pos, new_ball_vel);
    return;
    break;
  }

  //LOG_POL(2,<<"Tools: Model Dash: "<<dash_power);
  //LOG_POL(2,<<" old ball vel "<<old_ball_vel);

  // copying current state variables
  new_ball_vel = old_ball_vel;
  new_ball_pos = old_ball_pos;
  new_my_pos = old_my_pos;
  new_my_vel = old_my_vel;
  new_my_ang = old_my_ang;

  if(do_random){
    double r=GET_RANDOM(-rmax_ball,rmax_ball);
    new_ball_vel.addToX( r );
    r=GET_RANDOM(-rmax_ball,rmax_ball);
    new_ball_vel.addToY( r );
    r=GET_RANDOM(-rmax_player,rmax_player);
    new_my_vel.addToX( r );
    r=GET_RANDOM(-rmax_player,rmax_player);
    new_my_vel.addToY( r );
  }

  //step 1 : accelerate objects

  // me
  if(dash_power != 0){    
  												
  	double dir_as_deg = RAD2DEG(dash_dir); 			// die uebergebene Dir, In 0..360 !!!!!!
  	
  	dir_as_deg =   ServerOptions::dash_angle_step 
                 * rint(dir_as_deg / ServerOptions::dash_angle_step);	// Dir diskretisieren --> The dash direction is discretized by server::dash_angle_step
        if (dir_as_deg > 180) { dir_as_deg -= 360.0; } //nun in -180..180
  	
  	double dir_rate = 	( fabs( dir_as_deg ) > 90.0
                            ? ServerOptions::back_dash_rate 
                                - ( ( ServerOptions::back_dash_rate - ServerOptions::side_dash_rate )
                                                       * ( 1.0 - ( fabs( dir_as_deg ) - 90.0 ) / 90.0 ) )
                            : ServerOptions::side_dash_rate + ( ( 1.0 - ServerOptions::side_dash_rate )
                                                       * ( 1.0 - fabs( dir_as_deg ) / 90.0 ) )
						);
  	
  	double dir_rueck = DEG2RAD(dir_as_deg);
  	
    acc.setX( dash_power * cos(new_my_ang+dir_rueck) * modelled_player->dash_power_rate * modelled_player->effort * dir_rate );
    acc.setY( dash_power * sin(new_my_ang+dir_rueck) * modelled_player->dash_power_rate * modelled_player->effort * dir_rate );
    
    if (acc.getX() || acc.getY())
    {
      new_my_vel += acc ;
      if ((tmp = new_my_vel.norm()) > ServerOptions::player_speed_max)
        new_my_vel *= ( ServerOptions::player_speed_max / tmp) ;
    }
    
    //TG09: DBLOG_POL(0, << "dir_as_deg : " << dir_as_deg << " dir_rate : " << dir_rate << " acc.x : " << acc.x << " acc.y : " << acc.y);
    
  }

  // ball
  if(kick_power != 0){ 
    Vector ball_dist = new_ball_pos - new_my_pos;
    if(ball_dist.norm() <= modelled_player->kick_radius){
      
      double ball_angle = ball_dist.arg()-new_my_ang;
      ball_angle = Tools::get_angle_between_mPI_pPI(ball_angle);

      double ball_dist_netto = ball_dist.norm() - modelled_player->radius - ServerOptions::ball_size;
      
      /* THIS IS WRONG!!! 
      kick_power *= ServerOptions::kick_power_rate *
	(1 - 0.25*fabs(ball_angle)/PI - 0.25*ball_dist_netto/WSinfo::me->kick_radius);
      */

      kick_power *= ServerOptions::kick_power_rate *
	(1 - 0.25*fabs(ball_angle)/PI - 0.25*ball_dist_netto/(modelled_player->kick_radius - modelled_player->radius - ServerOptions::ball_size));
      
      acc.setX( kick_power * cos(kick_angle+new_my_ang) );
      acc.setY( kick_power * sin(kick_angle+new_my_ang) );
      
      if (acc.getX() || acc.getY()) {
	new_ball_vel += acc ;
	if ((tmp = new_ball_vel.norm()) > ServerOptions::ball_speed_max)
	  new_ball_vel *= (ServerOptions::ball_speed_max / tmp) ;
      }
    }
  }
  
  // turn me
  if(turn_angle != 0){
    new_my_ang += turn_angle/(1.0 + 1.0*new_my_vel.norm()* modelled_player->inertia_moment);
    new_my_ang = Tools::get_angle_between_mPI_pPI(new_my_ang);
  }

  //LOG_POL(2,<<"my old pos "<<new_my_pos<<" old ball pos "<<new_ball_pos
	//  <<" old ball vel "<<new_ball_vel);

  //step 2 : move
  new_my_pos += new_my_vel;     //me
  new_ball_pos += new_ball_vel; //ball

  //LOG_POL(2,<<"my new pos "<<new_my_pos<<" new ball pos "<<new_ball_pos
	  //<<" new ball vel "<<new_ball_vel);

  
  //step 3 : decay speed
  new_my_vel *= modelled_player->decay; //me
  new_ball_vel *= ServerOptions::ball_decay; //ball

}

/**
 * This method emulates an angular tackle command, as it may be
 * issued from Soccer Server V12 on.
 * 
 * Parameter checkAngle should be within [0,359]
 */
void
Tools::model_tackle_V12( const Vector & playerPos,
                         const ANGLE  & playerANG,
                         const Vector & ballPos,
                         const Vector & ballVel,
                         const int    & tackleAngleParameter,
                         Vector       & ballNewPos,
                         Vector       & ballNewVel)
{
  int tackleAngle = tackleAngleParameter % 360;
  if (tackleAngle < 0) tackleAngle += 360;
  //tackle power depends on the tackle angle
  double effectiveTacklePower
    =   ServerOptions::max_back_tackle_power
      +   (  ServerOptions::max_tackle_power
           - ServerOptions::max_back_tackle_power)
        * (fabs(180.0 - tackleAngle)/180.0);
  Vector player_2_ball = ballPos - playerPos;
  player_2_ball.rotate( - playerANG.get_value_mPI_pPI() );
  ANGLE player_2_ball_ANGLE = player_2_ball.ARG();
  effectiveTacklePower
    *= 1.0 - 0.5 * ( fabs( player_2_ball_ANGLE.get_value_mPI_pPI() ) / PI );

  //directed tackle: resulting ball velocity
  Vector tackle_dir = Vector( playerANG + ANGLE(DEG2RAD(tackleAngle)) );
  tackle_dir.normalize(   ServerOptions::tackle_power_rate 
                        * effectiveTacklePower);
  tackle_dir += ballVel;
  if ( tackle_dir.norm() > ServerOptions::ball_speed_max )
    tackle_dir.normalize( ServerOptions::ball_speed_max );
  //set result variables
  ballNewPos = ballPos + tackle_dir;
  ballNewVel = tackle_dir;
  ballNewVel *= ServerOptions::ball_decay;
}

void Tools::simulate_player(const Vector & old_pos, const Vector & old_vel, 
			    const ANGLE & old_ang, const int &old_stamina, 
			    const Cmd_Body & cmd,
			    Vector & new_pos, Vector & new_vel,
			    ANGLE & new_ang, int &new_stamina,
			    const double stamina_inc_max,
			    const double inertia_moment,
			    const double dash_power_rate, const double effort, const double decay){
  
  // simulate one step of an arbitrary player

  Vector acc;
  double tmp;

  /* computation of predicted state for ball and me */
  double turn_angle = 0;
  double dash_power = 0;
  new_stamina = old_stamina; // default: do not change

  switch(cmd.get_type()){
  case Cmd_Body::TYPE_DASH:
    cmd.get_dash(dash_power);
    if(dash_power>0)
      new_stamina += (int)(-dash_power + stamina_inc_max); 
    // assume that stamina increase is ok.
    else
      new_stamina += (int)(2*dash_power + stamina_inc_max); 
    break;
  case Cmd_Body::TYPE_TURN:
    cmd.get_turn(turn_angle);
    turn_angle = Tools::get_angle_between_mPI_pPI(turn_angle);
    break;
  }

  // copying current state variables
  new_pos = old_pos;
  new_vel = old_vel;
  new_ang = old_ang;

  //step 1 : accelerate objects

  // player
  if(dash_power != 0){    
    acc.setX( dash_power * cos(new_ang) * dash_power_rate * effort );
    acc.setY( dash_power * sin(new_ang) * dash_power_rate * effort );
    
    if (acc.getX() || acc.getY()) {
      new_vel += acc ;
      if ((tmp = new_vel.norm()) > ServerOptions::player_speed_max)
	new_vel *= ( ServerOptions::player_speed_max / tmp) ;
    }
  }
  // turn me
  if(turn_angle != 0){
    new_ang += ANGLE(turn_angle/(1.0 + 1.0*new_vel.norm()* inertia_moment)); 
    //    new_ang = new_ang.ANGLE(Tools::get_angle_between_mPI_pPI(new_ang)); // not needed here?
  }

  //step 2 : move
  new_pos += new_vel;     //me
  
  //step 3 : decay speed
  new_vel *= decay; //me
}


void Tools::get_successor_state (MyState const &state, Cmd_Body const &cmd, MyState &next_state,
                                 const bool do_random)
{
    Angle na;
    Angle a = state.my_angle.get_value_0_p2PI ();

    setModelledPlayer (state.me);
    model_cmd_main (state.my_pos, state.my_vel, a, state.ball_pos, state.ball_vel, cmd, next_state.my_pos,
                    next_state.my_vel, na, next_state.ball_pos, next_state.ball_vel, do_random);
    setModelledPlayer (WSinfo::me);
    next_state.me = state.me;
    next_state.my_angle = ANGLE (na);
    next_state.op_pos = state.op_pos;
    next_state.op_bodydir = state.op_bodydir;
    next_state.op_bodydir_age = state.op_bodydir_age;
    next_state.op = state.op;
}

bool Tools::intersection(const Vector & r_center, double size_x, double size_y,
                  const Vector & l_start, const Vector & l_end) {

  double p1_x= l_start.getX() - r_center.getX();
  double p1_y= l_start.getY() - r_center.getY();
  double p2_x= l_end.getX()   - r_center.getX();
  double p2_y= l_end.getY()   - r_center.getY();

  size_x *= 0.5;
  size_y *= 0.5;

  //now the rectangle is centered at (0,0) 

  double diff_x= (p2_x - p1_x);
  double diff_y= (p2_y - p1_y);

  if ( fabs(diff_x) >= 0.0001) {
    double N= (size_x - p1_x) / diff_x;
    double Y= p1_y + N * diff_y;
    if (0.0 <=  N  && N <= 1 && Y <= size_y && Y >= -size_y) return true;

    N= (-size_x - p1_x) / diff_x;
    Y= p1_y + N * diff_y;
    if (0.0 <=  N  && N <= 1 && Y <= size_y && Y >= -size_y) return true;
  }

  if ( fabs(diff_y) >= 0.0001) {
    double N= (size_y - p1_y) / diff_y;
    double X= p1_x + N * diff_x;
    if (0.0 <=  N  && N <= 1 && X <= size_x && X >= -size_x) return true;

    N= (-size_y - p1_y) / diff_y;
    X= p1_x + N * diff_x;
    if (0.0 <=  N  && N <= 1 && X <= size_x && X >= -size_x) return true;
  }
  return false;
}      

double Tools::ballspeed_for_dist(const double dist){
  double result;
  const double final_vel = 0.0; // final velocity after dist dist

  result = (1-ServerOptions::ball_decay)*(dist + final_vel*ServerOptions::ball_decay)
    + ServerOptions::ball_decay * final_vel;
  
  return result;
}

double Tools::get_ballspeed_for_dist_and_steps(const double dist, const int steps){
  
  double ballspeed = 1.0;
  double normdist = 0.;
  for(int i= 0; i<steps; i++ ){
    normdist += ballspeed;
    ballspeed *=ServerOptions::ball_decay;
  }
  // normdist is the distance that can be done in n steps.
  double result = dist/ normdist;
  //  LOG_MOV(0,"dist: "<<dist<<" steps "<<steps<<" normdist "<<normdist<<" ballspeed "<<ballspeed<<" result "<<result);

  return result;
}

int Tools::get_steps_for_ball_distance(const double dist,  double ballspeed){
  double tmp_dist = 0.;
  if(dist == 0)
    return 0;
  
  int i=0;
  do{
    tmp_dist += ballspeed;
    ballspeed *=ServerOptions::ball_decay;
    i ++;
  } while (tmp_dist < dist && i< 50);
  return i;
}


Vector Tools::ip_with_right_penaltyarea(const Vector p, const float dir){
// assumes that p is outside right penalty area; returns 0 if ip is not on right penalty area
  Vector dirvec;
  dirvec.init_polar(1.0, dir);
  double border_x, alpha;
  Vector ip;

  if(p.getX() > FIELD_BORDER_X - PENALTY_AREA_LENGTH)
    return Vector(0);

  if(dirvec.getX() >0)
    border_x = FIELD_BORDER_X - PENALTY_AREA_LENGTH; // test for opponent goal line
  else
    return Vector(0);

  // first check for horizontal line
  alpha = (border_x - p.getX()) / dirvec.getX();
  ip.setY( p.getY() + alpha * dirvec.getY() );
  ip.setX( border_x );
  if(fabs(ip.getY()) <= PENALTY_AREA_WIDTH/2. + 3.)
    return ip;  
  return Vector(0);
}



Vector Tools::ip_with_fieldborder(const Vector p, const float dir){
  Vector dirvec;
  dirvec.init_polar(1.0, dir);
  double border_x, border_y, alpha;
  Vector ip;

  //  LOG_POL(0,"check ip w fieldborder , pos: "<<p<<" dir "<<RAD2DEG(dir));

  if(dirvec.getX() >0)
    border_x = FIELD_BORDER_X; // test for opponent goal line
  else
    border_x = - FIELD_BORDER_X;

  // first check for horizontal line
  alpha = (border_x - p.getX()) / dirvec.getX();
  ip.setY( p.getY() + alpha * dirvec.getY() );
  ip.setX( border_x );
  if(fabs(ip.getY()) <= FIELD_BORDER_Y)
    return ip;  
  // then check for vertical line
    
  if(dirvec.getY() >0)
    border_y = FIELD_BORDER_Y; // test for left side
  else
    border_y = - FIELD_BORDER_Y;

  alpha = (border_y - p.getY()) / dirvec.getY();
  ip.setX( p.getX() + alpha * dirvec.getX() );
  ip.setY( border_y );
  return ip;
}


/* Berechnet den Schnittpunkt zweier Geraden */
Vector Tools::intersection_point(Vector p1, Vector steigung1, 
				 Vector p2, Vector steigung2) {
  double x, y, m1, m2;
  if ((steigung1.getX() == 0) || (steigung2.getX() == 0)) {
    if (fabs(steigung1.getX()) < 0.00001) {
      return point_on_line(steigung2, p2, p1.getX());
    } else if (fabs(steigung1.getX()) < 0.00001) {
      return point_on_line(steigung1, p1, p2.getX());
    } 
  }
  m1 = steigung1.getY()/steigung1.getX();
  m2 = steigung2.getY()/steigung2.getX();
  if (m1 == m2) return Vector(-51.5, 0);
  x = (p2.getY() - p1.getY() + p1.getX()*m1 - p2.getX()*m2) / (m1-m2);
  y = (x-p1.getX())*m1 + p1.getY();
  return Vector (x, y);

}



/* Berechnet die y-Koordinate Punktes auf der Linie, der die x-Koordinate x hat
 */
Vector Tools::point_on_line(Vector steigung, Vector line_point, double x) {
  //steigung.normalize();
  steigung = (1.0/steigung.getX()) * steigung;
  if (steigung.getX() > 0) {
    return (x - line_point.getX()) * steigung + line_point;
  }
  if (steigung.getX() < 0) {
    return (line_point.getX() - x) * steigung + line_point;
  }  // Zur Sicherheit, duerfte aber nie eintreten
  return line_point;
} /* point_on_line */


bool Tools::point_in_triangle(const Vector & p, const Vector & t1, const Vector & t2, const Vector & t3) {
  // look for  p= a* t1 + b * t2 + c * t3 with a+b+c=1 and a,b,c >= 0
  // if such a solution doesn't exits, then the point cannot be in the triangle;

  double A= t2.getX() - t1.getX();
  double B= t3.getX() - t1.getX();
  double C= t2.getY() - t1.getY();
  double D= t3.getY() - t1.getY();
  
  double det= A * D - C * B;
  if ( fabs(det) < 0.000001 ) {//consider matrix non regular (numerical stability)
    //cout << " false, det= " << det;
    return false;
  }

  double x= p.getX() - t1.getX();
  double y= p.getY() - t1.getY();
  

  double a= D * x - B * y;
  double b= -C * x + A * y;
  
  a/= det;
  b/= det;

  if (a < 0 || b < 0) {
    //cout << "\n false, a= " << a << " b= " << b;
    return false;
  }
  if ( a + b > 1.0) {
    //cout << "\n false, a= " << a << " b= " << b << " a+b= " << a+ b;
    return false;
  }

#if 0
  cout << "\n A= " << A << " B= " << B;
  cout << "\n C= " << C << " D= " << D;
  cout << "\n x= " << x << " y= " << y;
  cout << "\n true  a= " << a << " b= " << b << " c= " << 1- (a+ b) << " det= " << det;
#endif
  return true;
}

bool Tools::point_in_rectangle(const Vector & p, const Vector & r1, const Vector & r2, const Vector & r3, const Vector & r4) {
  if ( point_in_triangle(p,r1,r2,r3) )
    return true;
  if ( point_in_triangle(p,r1,r2,r4) )
    return true;
  if ( point_in_triangle(p,r1,r3,r4) )
    return true;
  if ( point_in_triangle(p,r2,r3,r4) )
    return true;
  return false;
}

bool Tools::point_in_field(const Vector & p, double extra_margin) {
  double border_x= FIELD_BORDER_X + extra_margin;
  double border_y= FIELD_BORDER_Y + extra_margin;
  return 
    p.getX() <= border_x   &&
    p.getX() >= -border_x  &&
    p.getY() <= border_y   &&
    p.getY() >= -border_y;
}

double Tools::get_ball_decay_to_the_power(int power) {
  if ( power <= 0 )
    return 1.0;

  if ( power < num_powers )
    return ball_decay_powers[power];

  int limit= MAX_NUM_POWERS;
  if ( power < MAX_NUM_POWERS ) //be as lazy as possible ;-)
    limit= power+1;

  if ( num_powers < 0) {
    ball_decay_powers[0]= 1.0;
    ball_decay_powers[1]= ServerOptions::ball_decay;
    num_powers= 2;
  }
  
  double start_value= ball_decay_powers[num_powers-1];
  while (num_powers < limit) {
    start_value *= ServerOptions::ball_decay;
    ball_decay_powers[num_powers]= start_value;
    num_powers++;
  }

  if ( power < num_powers ) 
    return ball_decay_powers[power];
 
  //the value of power is bigger then the size of the cache, so compute it directly
  INFO_OUT << "power= " << power << " > " 
	   << "MAX_NUM_POWERS= " << MAX_NUM_POWERS 
	   << " consider increasing the cache size";
  for ( int i= num_powers; i<= power; i++ )
    start_value *= ServerOptions::ball_decay;

  return start_value;
}

double Tools::get_max_expected_pointto_error( double dist )
{
  double sigma = pow(dist / 60, 4) * 178.25 + 1.75;
  return 2 * sigma;
}

/* view and neck stuff */

ANGLE Tools::get_view_angle_width(int vang) {
  if(!(WSinfo::ws->synch_mode_ok || Blackboard::get_guess_synched()))
  {
	  Angle normal = ServerOptions::visible_angle * PI/180.0;
	  if ( WIDE == vang )
		return ANGLE(normal*2.0);
	  if ( NARROW == vang )
		return ANGLE(normal*0.5); 
	  return ANGLE(normal);
  }
  else
  {
  	   Angle narrow = ServerOptions::visible_angle * ((ServerOptions::simulator_step * PI)  / (ServerOptions::send_step * 180.0));
       if ( NORMAL == vang )
         return ANGLE(narrow*2.0);
       if ( WIDE == vang )
       	 return ANGLE(narrow*3.0);
       // otherwise return narrow angle
       return ANGLE(narrow); 
  }
}

ANGLE Tools::cur_view_angle_width() {
  return get_view_angle_width(WSinfo::ws->view_angle);
}

ANGLE Tools::next_view_angle_width() {
  return get_view_angle_width(get_next_view_angle());
}

int Tools::get_next_view_angle() {
  return Blackboard::get_next_view_angle();
}

int Tools::get_next_view_quality() {
  return Blackboard::get_next_view_quality();
}

long Tools::get_last_lowq_cycle() {
  return Blackboard::get_last_lowq_cycle();
}

Vector Tools::get_last_known_ball_pos() {
  return Blackboard::get_last_ball_pos();
}

void Tools::force_highq_view() {
  if(WSinfo::ws->play_mode!=PM_PlayOn) Blackboard::force_highq_view=true;
}

void Tools::set_neck_request(int req_type, double param, bool force)
{
  if(Blackboard::neckReq.is_set()){
    if(force == false){
      LOG_POL(0,"Tools Error: Cannot set Neck Request; already set!");
      //LOG_ERR(0,"Tools Error: Cannot set Neck Request; already set!");
      return;
    }
    else{
      LOG_POL(0,"Tools WARNING: Neck request was already set, overwriting!");
    }
  }
  LOG_POL(0,"Tools NeckRequest: Success, Neck Request has been set ["<<req_type<<","<<param<<"]!");
  Blackboard::set_neck_request(req_type,param);
}

void Tools::set_neck_request(int req_type, ANGLE param, bool force) 
{
  Tools::set_neck_request(req_type, param.get_value(), force);
}

int Tools::get_neck_request(double &param) {
  return Blackboard::get_neck_request(param);
}

int Tools::get_neck_request(ANGLE &param) {
  double dum;
  int res = Blackboard::get_neck_request(dum);
  param = ANGLE(dum);
  return res;
}

void Tools::set_attention_to_request(int plNr, bool force)
{
  if(Blackboard::cvAttentionToRequest.is_set())
  {
    if (force == false)
    {
      LOG_POL(0,"Tools Error: Cannot set AttentionTo Request. It is already set ("<<Blackboard::cvAttentionToRequest.get_param()<<")!");
      LOG_ERR(0,"Tools Error: Cannot set AttentionTo Request. It is already set ("<<Blackboard::cvAttentionToRequest.get_param()<<")!");
      return;
    }
    else
    {
      LOG_POL(0,"Tools WARNING: AttentionTo request was already set ("
        <<Blackboard::cvAttentionToRequest.get_param()
        <<")! Overwriting: "<<plNr);
    }
  }
  LOG_POL(0,"Tools AttnetionToRequest: Success, AttentionTo Request has been set ["<<plNr<<"]!");
  Blackboard::set_attention_to_request(plNr);
}

int Tools::get_attention_to_request() 
{
  return Blackboard::get_attention_to_request();
}



bool Tools::is_ball_safe_and_kickable(const Vector &mypos, const Vector &oppos, const ANGLE &opbodydir,
				    const Vector &ballpos, int bodydir_age){
  if(mypos.distance(ballpos) > WSinfo::me->kick_radius)
    return false;
  return is_ballpos_safe(oppos,opbodydir,ballpos,bodydir_age);
}

/* extended version with hetero player support and tackles, taking kick_rand into account */
bool Tools::is_ball_safe_and_kickable(const Vector &mypos, const PPlayer opp, const Vector &ballpos,
				      bool consider_tackles) {

  
  double ball_safety=1.41*(WSinfo::ball->vel.norm()*WSinfo::me->kick_rand_factor
			  +ServerOptions::ball_rand*ServerOptions::ball_speed_max);
  if(mypos.distance(ballpos) > WSinfo::me->kick_radius-ball_safety){ 
    //LOG_MOV(0,"Checking new ballpos safety: Next ball position not in kickrange of player");
    return false;
  }
  return is_ballpos_safe(opp,ballpos,consider_tackles);
}

bool Tools::is_ball_safe_and_kickable(const Vector &mypos, const PlayerSet &opps, const Vector &ballpos,
				      bool consider_tackles) {
  for(int p=0;p<opps.num;p++)
    if(! is_ball_safe_and_kickable(mypos,opps[p],ballpos,consider_tackles)) return false;
  return true;
}

bool Tools::is_ballpos_safe(const Vector &oppos, const ANGLE &opbodydir,
			    const Vector &ballpos, int bodydir_age){
  // returns true if ballpos is safe in next time step, if an opponent approaches in worst case
#define SAFETY .25

  if(bodydir_age >0){ // if I'm not sure about age, take worst case
#if 0
    LOG_MOV(0,"Age of body dir "<<bodydir_age<<" dist2 ball " <<oppos.distance(ballpos)
	    <<" critical range "<<ServerOptions::kickable_area + ServerOptions::player_speed_max + SAFETY);
#endif
    if(oppos.distance(ballpos) < ServerOptions::kickable_area + ServerOptions::player_speed_max + SAFETY)
      return false;
    else
      return true;
  }

  // idea: rotate the opponent so that it virtually accelerates along the x-Axis
  // rotate ballpos, so that the relative Position remains (new_center)
  Vector new_center = ballpos - oppos;
  new_center.rotate(-(opbodydir.get_value())); 

  float op_radius = ServerOptions::kickable_area; // ridi: could be refined by exact op data
  float op_speed = ServerOptions::player_speed_max; // ridi: could be refinedby exact op data

  float y_thresh = op_radius + SAFETY;
  float x_thresh = op_radius + op_speed + SAFETY;

  //#define DRAW

  if(new_center.getY() > y_thresh){
#if DRAW
    LOG_MOV(0,_2D<<L2D(-( op_speed + op_radius), op_radius,(op_speed + op_radius), op_radius, "#ffff00"));
    LOG_MOV(0,_2D<<C2D(new_center.x,new_center.y,.1,"#ffff00"));
    LOG_MOV(0,<<"ok, y larger");
#endif
  /* Sput: This won't work as expected, since here we assume the opp has a quadrangular kickrange...
           so this routine is more conservative than necessary.
	   On the other hand it is quite fast and we are dealing with estimated values anyway...

	   Note that another ballpos_safe(), dealing with heteros and tackles, can be found below!
  */
    return true;
  }
  if(new_center.getY() < - y_thresh){
#if DRAW
    LOG_MOV(0,_2D<<L2D(-( op_speed + op_radius), -op_radius,(op_speed + op_radius), -op_radius, "#ffff00"));
    LOG_MOV(0,_2D<<C2D(new_center.x,new_center.y,.1,"#ffff00"));
    LOG_MOV(0,<<"ok, y smaller");
#endif
    return true;
  }
  if(new_center.getX()  > x_thresh){
#if DRAW
    LOG_MOV(0,_2D<<L2D(( op_speed + op_radius), -op_radius,(op_speed + op_radius), op_radius, "#ffff00"));
    LOG_MOV(0,_2D<<C2D(new_center.x,new_center.y,.1,"#ffff00"));
    LOG_MOV(0,<<"ok, x larger");
#endif
    return true;
  }
  if(new_center.getX()  < - x_thresh){
#if DRAW
    LOG_MOV(0,_2D<<L2D(-( op_speed + op_radius), -op_radius,-(op_speed + op_radius), op_radius, "#ffff00"));
    LOG_MOV(0,_2D<<C2D(new_center.x,new_center.y,.1,"#ffff00"));
    LOG_MOV(0,<<"ok, x smaller");
#endif
    return true;
  }
  return false;
}

/* use real opp data (hetero players...) and if wanted, also consider tackles! */
/* Also considers the goalie.                                                  */
#define NEW_BALLPOS_SAFE
#define NEW_SAFETY .01
#define BALL_MAX_SAFETY .12
#define PLAYER_MAX_SAFETY .10

/* this one takes a complete player set and tests every opponent.
   You SHOULD take care that only necessary players are within the pset - this routine
   does NOT remove players from it!
*/
bool Tools::is_ballpos_safe(const PlayerSet &opps,const Vector &ballpos,bool consider_tackles) {
  for(int p=0;p<opps.num;p++)
    if(! is_ballpos_safe(opps[p],ballpos,consider_tackles)) return false;
  return true;
}

bool Tools::is_ballpos_safe(const PPlayer opp,const Vector &ballpos,bool consider_tackles) {
  int minInact, maxInact;
  WSinfo::get_player_inactivity_interval( opp, minInact, maxInact );
  if (minInact > 0) return true;

  double tackle_dist_threshold = 1.5; // this corresponds to a tackle probabilty of less than 83 %

  //  if(consider_tackles)
  //  LOG_MOV(0,<<"Enter NEW check ballpos! consider_tackles : "<<consider_tackles); 

  //bool is_goalie=(WSinfo::ws->his_goalie_number>0&&opp->number==WSinfo::ws->his_goalie_number)?
  //  true:false;
  bool is_goalie=(WSinfo::his_goalie && WSinfo::his_goalie==opp);
  //LOG_POL(0,<<"ballpos_safe: is_goalie="<<is_goalie);
  //LOG_POL(0,<<"my effort="<<WSinfo::me->effort);
  double ball_safety=min(.5*1.41*(ServerOptions::ball_rand*ServerOptions::ball_speed_max
			  +WSinfo::me->kick_rand_factor*WSinfo::ball->vel.norm()),
			BALL_MAX_SAFETY);
  double opp_maxspeed,opp_maxspeed_back;
  //LOG_POL(0,<<"age_vel="<<opp->age_vel<<", effort="<<opp->effort);
  if(opp->age_vel>0) { // we don't know opp vel -> worst case!
    opp_maxspeed=opp->speed_max;
    opp_maxspeed_back=opp->speed_max;
  } else {
    opp_maxspeed=min(opp->vel.norm()+(100.0*opp->dash_power_rate*opp->effort),
		     opp->speed_max);
    opp_maxspeed_back=min(-opp->vel.norm()+(100.0*opp->dash_power_rate*opp->effort),
			  opp->speed_max);
  }
  if(opp->age_ang>0) { // we don't know opp dir -> worst case!
    
    double max_radius;
    if(is_goalie) max_radius=ServerOptions::catchable_area_l+opp->speed_max;
    else {
      if(consider_tackles) {
	max_radius=tackle_dist_threshold +opp_maxspeed;
      } else {
	max_radius=opp->kick_radius+opp_maxspeed;
      }
    }
    max_radius+= min(.5*1.41*ServerOptions::player_rand*opp->speed_max,PLAYER_MAX_SAFETY);
    if(opp->pos.distance(ballpos) < max_radius+ball_safety+NEW_SAFETY) {
      //      LOG_MOV(0,<<"NEW check ballpos: Not safe [worst case assumption, op bodydir not known!]");
      return false;
    }
    else{
      //  LOG_MOV(0,<<"NEW check ballpos: Save  [worst case assumption, op bodydir not known!]");
      return true;
    }
  } // ok, current body dir data available, so consider opp's body dir!
  Vector new_center = ballpos - opp->pos;
  new_center.rotate(-(opp->ang.get_value()));

  double op_radius;

  if(is_goalie) {
    op_radius=ServerOptions::catchable_area_l;
  } 
  else {
    op_radius=opp->kick_radius;
  }
  
  double nodash_safety=min( .5*ServerOptions::player_rand*opp->vel.norm(),PLAYER_MAX_SAFETY);
  double dash_safety=min( .5*1.41*ServerOptions::player_rand*opp_maxspeed,PLAYER_MAX_SAFETY);

    if(consider_tackles) {
      if(new_center.getX()>=0){ // ballpos is in the direction of the opponent's bodydir
       	if((new_center.getX()< opp_maxspeed+op_radius + tackle_dist_threshold +dash_safety+ball_safety+NEW_SAFETY) &&
	   //   (fabs(new_center.y < 0.7))) // ridi05: discovered in Osaka; this does not work
	   (fabs(new_center.getY()) < 0.7))  // this was the version used in the finals; but is t correct???
	  return false;
      } // can be tackled by quickly moving
      if(new_center.norm() < tackle_dist_threshold){
	return false; // ball can be reached by turning !
      }
    }


  if(new_center.getY()> op_radius+nodash_safety+ball_safety+NEW_SAFETY) return true;
  if(new_center.getY()<-op_radius-nodash_safety-ball_safety-NEW_SAFETY) return true;
  if(new_center.getX()> opp_maxspeed+op_radius+dash_safety+ball_safety+NEW_SAFETY) return true;
  if(new_center.getX()<-opp_maxspeed_back-op_radius-dash_safety-ball_safety-NEW_SAFETY) return true;
  return false;
}


bool Tools::is_position_in_pitch(Vector position,  const float safety_margin){
  if ( (position.getX()-safety_margin < -ServerOptions::pitch_length / 2.)
       || (position.getX()+safety_margin > ServerOptions::pitch_length / 2.) ) return false;
  if ( (position.getY()-safety_margin < -ServerOptions::pitch_width / 2.)
       || (position.getY()+safety_margin > ServerOptions::pitch_width / 2.) ) return false;
  return true;
}


double Tools::min_distance_to_border(const Vector position){
  return MIN(ServerOptions::pitch_length / 2. - fabs(position.getX()), ServerOptions::pitch_width / 2. - fabs(position.getY()));
}



bool Tools::is_a_scoring_position(Vector pos){
  const XYRectangle2d hot_scoring_area( Vector(FIELD_BORDER_X - 8, -7),
					Vector(FIELD_BORDER_X, 7)); 

  double dist2goal = (Vector(FIELD_BORDER_X,0) - pos).norm();
  if(dist2goal > 20.0){
    //DBLOG_POL(0,"CHECK SCORING POS "<<pos<<" Failure! dist2goal > 25: "<<dist2goal);
    return false;
  }
  
  PlayerSet pset = WSinfo::valid_opponents;
  Vector opgoalpos =Vector(FIELD_BORDER_X,0.);
  double scanrange = 8.0;
  Quadrangle2d check_area = Quadrangle2d(pos,opgoalpos,scanrange/2., scanrange);
  //DBLOG_DRAW(0,check_area);
  pset.keep_players_in(check_area);
  if(pset.num == 0){
    //DBLOG_POL(0,"CHECK SCORING POS: Area  -> SUCCESS: empty ");
    return true;
  }
  if(pset.num == 1 && pset[0] == WSinfo::his_goalie){
    //DBLOG_POL(0,"CHECK SCORING POS: Area  -> SUCCESS: only goalie before position ");
      return true;
  }
  if(dist2goal > 15.0){
    //DBLOG_POL(0,"CHECK SCORING POS: FAILURE too far away for direct shot ");
      return false;
  }
  pset = WSinfo::valid_opponents;
  scanrange = 4.0;
  check_area = Quadrangle2d(pos,opgoalpos,scanrange, scanrange);
  //DBLOG_DRAW(0,check_area);
  pset.keep_players_in(check_area);
  if(pset.num == 0){
    //DBLOG_POL(0,"CHECK SCORING POS: Area  -> SUCCESS: can score to middle ");
    return true;
  }

  pset = WSinfo::valid_opponents;
  check_area = Quadrangle2d(pos,Vector(FIELD_BORDER_X,+6.),scanrange, scanrange);
  //DBLOG_DRAW(0,check_area);
  pset.keep_players_in(check_area);
  if(pset.num == 0){
    //DBLOG_POL(0,"CHECK SCORING POS: Area  -> SUCCESS: can score to right ");
    return true;
  }

  pset = WSinfo::valid_opponents;
  check_area = Quadrangle2d(pos,Vector(FIELD_BORDER_X,-6.),scanrange, scanrange);
  //DBLOG_DRAW(0,check_area);
  pset.keep_players_in(check_area);
  if(pset.num == 0){
    //DBLOG_POL(0,"CHECK SCORING POS: Area  -> SUCCESS: can score to left ");
    return true;
  }
  

  if(hot_scoring_area.inside(pos)){
    //DBLOG_POL(0,"CHECK SCORING POS "<<pos<<": inside scoring position");
    return true;
  }
  //DBLOG_POL(0,"CHECK SCORING POS "<<pos<<": FAILURE not a scoring position");
  return false;
}



bool Tools::shall_I_wait_for_ball(const Vector ballpos, const Vector ballvel, int &steps){
#define SAFETY_MARGIN .3
#define MAX_STEPS 20
  steps = -1; // init
  const XYRectangle2d hot_scoring_area( Vector(FIELD_BORDER_X - 14, -11),
					Vector(FIELD_BORDER_X, 11)); 


  if(ballpos.getX() > WSinfo::me->pos.getX() &&
     hot_scoring_area.inside(WSinfo::me->pos) == false) // ball comes from before me -> go against it
    return false;
  PlayerSet pset= WSinfo::valid_opponents;
  pset.keep_players_in_circle(WSinfo::me->pos, 3.);
  if(pset.num > 0){
    // LOG_POL(0,"Ball is coming right to me, and no op. close to me -> just face ball");
    return false;
  }

  Vector predicted_ballpos = ballpos;
  Vector predicted_ballvel = ballvel;
  for(int i=0;i<MAX_STEPS;i++){
    predicted_ballpos += predicted_ballvel;
    predicted_ballvel *= ServerOptions::ball_decay;
    if(WSinfo::me->pos.distance(predicted_ballpos)<WSinfo::me->kick_radius){
      steps = i;
      break;
    }
  }
  if(steps <= 2){ // Ball is close2player
    if(WSinfo::me->pos.distance(predicted_ballpos)<WSinfo::me->kick_radius - SAFETY_MARGIN){
      LOG_POL(0,<<"Ball needs 2 or less steps and I'll get it without moving");
      return true;
    }
    return false; // Ball is close but I might miss it -> start active intercepting
  }

  return steps > 0; // I'll get the ball in less than maxsteps
}
  
double Tools::speed_after_n_cycles(const int n, const double dash_power_rate,
				  const double effort, const double decay){
  // returns actual speed after n cycles when starting w. speed 0, and dashing 100
  double u = 0;
  double v= 0;
  for(int i= 0; i<n; i++){
    u = v + 100. * dash_power_rate * effort;
    v = u * decay;
  }
  //  LOG_POL(0,"Tools speed after "<<n<<" cycles, power rate "<<dash_power_rate<<" decay "<<decay<<" speed: "<<u);
  return u;
}

double Tools::eval_pos(const Vector & pos, const double mindist2teammate ){
  // ridi 04: simple procedure to evaluate quality of position: 0 bad 1 good

  PPlayer tcp; // teammate closest to ball
  PlayerSet pset = WSinfo::valid_teammates_without_me;
  pset.keep_players_in_circle(WSinfo::ball->pos, 8.0); // consider only players in reasonable distance to the ball
  tcp = pset.closest_player_to_point(WSinfo::ball->pos);
  if(tcp == 0)
    return 1.0;  // no player found that is closest to ball; return position is ok.
  return eval_pos_wrt_position(pos,WSinfo::ball->pos, mindist2teammate);
}


double Tools::eval_pos_wrt_position(const Vector & pos,const Vector & targetpos, const double mindist2teammate){
  // ridi 04: simple procedure to evaluate quality of position: 0 bad 1 good
  double result = 0.0;

  PlayerSet pset = WSinfo::valid_opponents;
  PPlayer closest_op= pset.closest_player_to_point(pos);
  double closest_op_dist;

  if(closest_op != NULL)
    closest_op_dist= pos.distance(closest_op->pos);
  else
    closest_op_dist= 1000;

  if(closest_op_dist < 2.0)
    return 0.0;

  // do not be too close to teammates
  pset = WSinfo::valid_teammates_without_me;
  pset.keep_players_in_circle(pos, mindist2teammate);
  if(pset.num > 0){
    result = 0.0;
    return result;
  }

  // check broader passway
  float width1 =  .9 * 2* ((pos-targetpos).norm()/2.5);
  float width2 = 4; // at ball be  a little smaller
  Quadrangle2d check_area = Quadrangle2d(pos, targetpos , width1, width2);
  LOG_POL(0, <<_2D<<check_area );
  pset = WSinfo::valid_opponents;
  pset.keep_players_in(check_area);
  if(pset.num == 0){
    result = 4.0 + Tools::min(closest_op_dist,3.0);
    return result;
  }
  // check smaller passway
  width1 =  .7 * 2* ((pos-targetpos).norm()/2.5);
  width2 = 3; // at ball be  a little smaller
  check_area = Quadrangle2d(pos, targetpos , width1, width2);
  LOG_POL(0, <<_2D<<check_area );
  pset.keep_players_in(check_area);
  if(pset.num == 0){
    result = 1.0  + Tools::min(closest_op_dist,3.0);
  }
  else
    result = 0.0;
  return result;
}


double Tools::evaluate_wrt_position(const Vector & pos,const Vector & targetpos){
  // ridi 04: simple procedure to evaluate quality of position: 0 bad 1 good 2 better
  double result = 0.0;

  // check broader passway
  float width1 =  1.0 * 2* ((pos-targetpos).norm()/2.5);
  float width2 = 4; // at ball be  a little smaller
  Quadrangle2d check_area = Quadrangle2d(pos, targetpos , width1, width2);
  //LOG_POL(0, <<_2D<<check_area );
  PlayerSet pset = WSinfo::valid_opponents;
  pset.keep_players_in(check_area);
  if(pset.num == 0){
    double mindist2op = min((pos-targetpos).norm()/2.5, 3.0);
    if( get_closest_op_dist(pos) > mindist2op){
      return 2.0;
    }
    return 1.0; // passway is free, but pos is close 2 op
  }
  // check smaller passway
  width1 =  .7 * 2* ((pos-targetpos).norm()/2.5);
  width2 = 3; // at ball be  a little smaller
  check_area = Quadrangle2d(pos, targetpos , width1, width2);
  //LOG_POL(0, <<_2D<<check_area );
  pset.keep_players_in(check_area);
  if(pset.num == 0){
    result = 1.0 ;
  }
  else
    result = 0.0;
  return result;
}

bool Tools::is_pos_free(const Vector & pos){
  PlayerSet pset = WSinfo::valid_teammates_without_me;
  pset.keep_players_in_circle(WSinfo::ball->pos, 3.0); // consider only players in reasonable distance to the ball
  pset.keep_players_in_circle(pos, 8.0); // consider only players in reasonable distance to the ball
  if(pset.num > 0)
    return false;
  return true;
}



double Tools::get_closest_op_dist(const Vector pos){
    PlayerSet pset = WSinfo::valid_opponents;
    PPlayer closest_op= pset.closest_player_to_point(pos);
    double closest_op_dist;
    if(closest_op != NULL)
      closest_op_dist= pos.distance(closest_op->pos);
    else
      closest_op_dist= 1000.;
    return closest_op_dist;
}

int Tools::num_teammates_in_circle(const Vector pos, const double radius){
    PlayerSet pset = WSinfo::valid_teammates_without_me;
    pset.keep_players_in_circle(pos,radius);
    return pset.num;
}


double Tools::get_closest_teammate_dist(const Vector pos){
    PlayerSet pset = WSinfo::valid_teammates_without_me;
    PPlayer closest_teammate= pset.closest_player_to_point(pos);
    double closest_teammate_dist;
    if(closest_teammate != NULL)
      closest_teammate_dist= pos.distance(closest_teammate->pos);
    else
      closest_teammate_dist= 1000.;
    return closest_teammate_dist;
}

double Tools::get_optimal_position(Vector & result, Vector * testpos,
				   const int num_testpos,const PPlayer &teammate){

   double mindist2teammate = 4.0;

  PPlayer ballholder; // teammate closest to ball
  Vector ipos; // not used, but needed for query
  int time2ball = -1;
  ballholder = get_our_fastest_player2ball(ipos, time2ball);
  if(ballholder != NULL){ // ballholder found and cycles <3. check it isn't me!!
    //DBLOG_DRAW(0, C2D(ballholder->pos.x,ballholder->pos.y,1.5,"blue"));
  }
  if(time2ball<0 || time2ball >=6){ // critcical value!!! 5 works, less than 5 is worse!!!
    //    DBLOG_POL(0,"CHECK BALLHODLER; NO BALLHOLDER FOUND OR TOO FAR. cycles need: "<<time2ball);
    ballholder = NULL; // reset
  }
  else if(ballholder != NULL){ // ballholder found and cycles <3. check it isn't me!!
    if(ballholder ->number == WSinfo::me->number){
      // I will be the next ballholder
      ballholder = NULL;
    }
  }


  result = testpos[0]; // default: use corrected target position.

  // now check, if I should find a better position within my region if I could probably get a pass

  double max_V=-1;
  int bestpos = 0; // default
  int bestpos_teamdist = 0;
  double max_Vteamdist = -1;


  for(int i= 0; i< num_testpos;i++){
    if(testpos[i].getY() >= ServerOptions::pitch_width/2. -1. ) // too far left
      testpos[i].setY( ServerOptions::pitch_width/2. -1. );
    if(testpos[i].getY() <= -(ServerOptions::pitch_width/2. -1.) ) // too far right
      testpos[i].setY( -(ServerOptions::pitch_width/2. -1.) );
    //    DBLOG_DRAW(0, C2D(testpos[i].x,testpos[i].y,0.3,"red")); // draw in any case, overdraw probably later

    
    double closest_op_dist = get_closest_op_dist(testpos[i]);
    double closest_teammate_dist = get_closest_teammate_dist(testpos[i]);

    if(closest_op_dist < 2.0)
      continue;
    if(closest_teammate_dist < mindist2teammate)
      continue;
    double Vteamdist = min(5.0,closest_teammate_dist);
    if(Vteamdist>max_Vteamdist){
      bestpos_teamdist = i;
      max_Vteamdist = Vteamdist;
    }

    double Vball = 0.0;
    if(ballholder != NULL){
      //      DBLOG_DRAW(0, C2D(ballholder->pos.x,ballholder->pos.y,2.0,"blue"));
      Vball = evaluate_wrt_position(testpos[i],ballholder->pos);
    }
    double Vteammate= 0.0;
    if(ballholder !=NULL && teammate != NULL && Vball == 0 && max_V < 1000) {
      //someone is close 2 ball and teammate exists and I have not found a position2ball yet
      //      DBLOG_DRAW(0, C2D(teammate->pos.x,teammate->pos.y,2.0,"blue"));
      Vteammate = evaluate_wrt_position(testpos[i],teammate->pos);
    }
    
    //    double V=Vopdist + 100 * Vteammate + 1000*Vball;
    double V= 100 * Vteammate + 1000*Vball;
    if(V>max_V){
      bestpos = i;
      max_V = V;
    }
    if(V>=1000.){
      //      DBLOG_DRAW(0, C2D(testpos[i].x,testpos[i].y,0.3,"green"));
    }
    else if(V>=100.){
      // DBLOG_DRAW(0, C2D(testpos[i].x,testpos[i].y,0.3,"blue"));
    }
    else if(V> 0.){
      //DBLOG_DRAW(0, C2D(testpos[i].x,testpos[i].y,0.3,"orange"));
    }
    //DBLOG_POL(0,"Evaluation pos  "<<i<<" "<<testpos[i]<<"  : "<< V);
  } // for all positions


  if(max_V <10.){
    result = testpos[bestpos_teamdist];
    //DBLOG_POL(0,"only found a position to keep distance from my teammate:  "<< result);
  }
  else{
    result = testpos[bestpos];
    //DBLOG_POL(0,"Found a position to run free:  "<< result);
  }

  //DBLOG_DRAW(0, C2D(result.x, result.y ,1.5,"red")); 
  //DBLOG_DRAW(0, C2D(result.x, result.y ,1.4,"blue")); 
  //DBLOG_POL(0,"Search Best position:  "<< result);
  if(WSinfo::me->pos.distance(result) < 1.0){
    result = WSinfo::me->pos;
  }  
  return max_V;
}


#define INFINITE_STEPS 1000

PPlayer Tools::get_our_fastest_player2ball(Vector &intercept_pos, int & steps2go){
  Vector ballpos = WSinfo::ball->pos;
  Vector ballvel = WSinfo::ball->vel;
#if 0
  double speed = WSinfo::ball->vel.norm();
  double dir = WSinfo::ball->vel.arg();
#endif 

  int myteam_fastest = INFINITE_STEPS;
  Vector earliest_intercept = Vector(0);
  PPlayer teammate = NULL;

  // check my team

  steps2go = -1;

  PlayerSet pset = WSinfo::valid_teammates;
  for (int idx = 0; idx < pset.num; idx++) {
    int my_time2react = 0;
    Vector player_intercept_pos;
    if(pset[idx]->pos.distance(ballpos) < 2.0){
      steps2go = 0; // this teammate has the ball!
      intercept_pos = pset[idx] ->pos;
      return pset[idx];
    }
    int steps = Policy_Tools::get_time2intercept_hetero(player_intercept_pos, ballpos, ballvel,
							pset[idx],
							my_time2react,myteam_fastest); 
    if(steps <0) 
      steps =INFINITE_STEPS; // player doesnt get ball in reasonable time
    if(steps < myteam_fastest){ // the fastest own player yet
      myteam_fastest = steps;
      earliest_intercept = player_intercept_pos;
      teammate=pset[idx];
    }
  }
  intercept_pos = earliest_intercept;
  
  if(myteam_fastest>=INFINITE_STEPS){
    steps2go = -1;
    return NULL;
  }
  
  steps2go = myteam_fastest;
  return teammate;
}

bool Tools::is_pos_occupied_by_ballholder(const Vector& pos){
   int time2ball = -1;
   Vector ipos;
   PPlayer ballholder = get_our_fastest_player2ball(ipos, time2ball);
   if(ballholder == NULL)
     return false;

   return time2ball < 4 && pos.distance(ipos) < 5.0;
}


bool Tools::is_ball_kickable_next_cycle (const Cmd &cmd, Vector &mypos, Vector &myvel, ANGLE &newmyang,
                                         Vector &ballpos, Vector &ballvel)
{
    model_cmd_main (WSinfo::me->pos, WSinfo::me->vel, WSinfo::me->ang, WSinfo::ball->pos,
                    WSinfo::ball->vel, cmd.cmd_body, mypos, myvel, newmyang, ballpos, ballvel);

    return ballpos.distance (mypos) <= WSinfo::me->kick_radius;
}


/***************************************************************/
 /* ridi 05: evaluate positions                                 */
 /***************************************************************/


Vector Tools::check_potential_pos(const Vector pos, const double max_advance){
  PlayerSet pset;
  Vector endofregion;
  double check_length = 30.;
  double endwidth = 1.66 * check_length;
  
  double startwidth = 5.;
  
  ANGLE testdir[10];
  int num_dirs = 0;
  testdir[num_dirs ++] = ANGLE(0);
  testdir[num_dirs ++] = ANGLE(45/180.*PI);
  testdir[num_dirs ++] = ANGLE(-45/180.*PI);
  testdir[num_dirs ++] = ANGLE(90/180.*PI);
  testdir[num_dirs ++] = ANGLE(-90/180.*PI);
  
  double max_evaluation = evaluate_pos(pos); // default: evaluate current position
  Vector best_potential_pos = pos; 

  for(int i=0; i<num_dirs; i++){
    pset = WSinfo::valid_opponents;
    endofregion.init_polar(check_length, testdir[i]);
    endofregion += pos;
    Quadrangle2d check_area = Quadrangle2d(pos, endofregion, startwidth, endwidth);
    //    LOG_POL(1,<<_2D<< check_area );
    pset.keep_players_in(check_area);
    double can_advance = check_length;
    if(pset.num >0){
      if(pset.num > 1 || pset[0] != WSinfo::his_goalie){
	// there is a player, can_advance is restricted to the dist to that player.
	PPlayer closest_op = pset.closest_player_to_point(pos);
	can_advance = pos.distance(closest_op->pos)*0.5;  // 0.5: potential opponent runs towards me
      }
    }
    // from here on, we know that we can advance in this direction!
    Vector potential_pos;
    can_advance = MIN(max_advance,can_advance);
    potential_pos.init_polar(can_advance,testdir[i]);
    potential_pos += pos;
    if(potential_pos.getX() > FIELD_BORDER_X)
      potential_pos.setX(FIELD_BORDER_X);
    if(potential_pos.getY() > FIELD_BORDER_Y)
      potential_pos.setY(FIELD_BORDER_Y);
    if(potential_pos.getY() < -FIELD_BORDER_Y)
      potential_pos.setY(-FIELD_BORDER_Y);
    //    LOG_POL(0,<<_2D<< C2D(potential_pos.x, potential_pos.y,2.0,"orange"));
    double evaluation = evaluate_pos(potential_pos);
    //    LOG_POL(0,<<"test potential "<<potential_pos<<" Evaluation: "<<evaluation<<" max_eval "<<max_evaluation);
    if(evaluation >max_evaluation){
      max_evaluation = evaluation;
      best_potential_pos = potential_pos;
    }
  }

  LOG_POL(0,<<_2D<< VC2D(best_potential_pos,1.7,"red"));
  return best_potential_pos;
}

double Tools::evaluate_pos_selfpass_neuro(const Vector query_pos){
  double evaluation;

  if(WSinfo::his_goalie == NULL) // goalie is not known
    return evaluate_pos_analytically(query_pos);

  if(WSinfo::his_goalie->pos.getY() >5.)// no chance to score there, prefer other direction
    return (-query_pos.getY());   // the more negativ, the better
  if(WSinfo::his_goalie->pos.getY() <-5.)// no chance to score there, prefer other direction
    return (query_pos.getY());   // the more positiv, the better
  // goalie is in between, keep going in your direction (Hysterese)
  if(WSinfo::me->pos.getY() > WSinfo::his_goalie->pos.getY()) // keep going to the left!
    return (query_pos.getY());

  // (WSinfo::me->pos.y <= WSinfo::goalie->pos.y) // keep going to the right
  return (-query_pos.getY());


  // not used...
  if(state_valid_at != WSinfo::ws->time){
    AbstractMDP::copy_mdp2astate(state);  // ridi05: copy should be rewritten and use WSinfo instead of mdpstate
    LOG_POL(0,"1vs1: Copy mdpstate in this cycle");
    state_valid_at = WSinfo::ws->time;
  }
  state.my_team[state.my_idx].pos = query_pos;
  state.ball.pos = query_pos + Vector(0.3,0.);
  evaluation = Planning::evaluate_byJnn_1vs1(state);
  if(evaluation >= 0){// neural evaluation
    evaluation = 90. - (90.-(-50.)) * evaluation;
    LOG_POL(0,"1vs1: Neural Evaluation of query pos  "<<query_pos<<": "<<evaluation);
    return evaluation;
  }
  else{
    LOG_POL(0,"1vs1: Standard Evaluation of query pos  "<<query_pos);
    return evaluate_pos_analytically(query_pos);
  }
}


double Tools::evaluate_pos_selfpass_neuro06(const Vector query_pos){
  double evaluation;

  if(state_valid_at != WSinfo::ws->time){
    AbstractMDP::copy_mdp2astate(state);  // ridi05: copy should be rewritten and use WSinfo instead of mdpstate
    LOG_POL(0,"1vs1: Copy mdpstate in this cycle");
    state_valid_at = WSinfo::ws->time;
  }


  Vector orig_ballpos = state.ball.pos;
  Vector orig_playerpos = state.my_team[state.my_idx].pos;
  state.my_team[state.my_idx].pos = query_pos;
  state.ball.pos = query_pos + Vector(0.3,0.);
  evaluation = Planning::evaluate_byJnn_1vs1(state);
  //  display_astate();
  state.ball.pos = orig_ballpos;
  state.my_team[state.my_idx].pos = orig_playerpos;

  if(evaluation >= 0){// neural evaluation
    evaluation = 100 * (1- evaluation);
    LOG_POL(0,"1vs1: Selfpass: Neural Evaluation of query pos  "<<query_pos<<": "<<evaluation);
    return evaluation;
  }
  else{
    LOG_POL(0,"1vs1: Standard Evaluation of query pos  "<<query_pos);
    return evaluate_pos_analytically(query_pos);
  }
}





int Tools::get_interceptor_in_astate(AState & state){
  double closest_dist = 1E6;
  int closest = -1;

  for(int i=0;i<11;i++){
    double tmp_dist = state.my_team[i].pos.sqr_distance(state.ball.pos);
    //    DBLOG_POL(0,"tmp_dist: "<<tmp_dist<<" closest dist "<<closest_dist<<" closet "<<closest);
    if(tmp_dist < closest_dist){
      closest_dist =  tmp_dist;
      closest = i;
    }
  }
  return closest;
}

void Tools::display_astate(){
  return;

  if(state_valid_at != WSinfo::ws->time){
    DBLOG_POL(0,"Cant display state. Not up to date");
    return;
  }

  for(int i=0;i<11;i++){
    DBLOG_DRAW(0,VC2D(state.my_team[i].pos,2.0,"blue"));
    DBLOG_DRAW(0,VC2D(state.op_team[i].pos,2.0,"red"));
  }
  DBLOG_DRAW(0,VC2D(state.ball.pos,1.5,"orange"));
}


double Tools::evaluate_pos_analytically(const Vector query_pos){
  double evaluation;

  if(query_pos.getX() <35.){
    evaluation = query_pos.getX();
  }
  else{
    evaluation = query_pos.getX() + FIELD_BORDER_Y - fabs(query_pos.getY());
  }
  return evaluation;
}

double Tools::evaluate_pos(const Vector query_pos){
  double evaluation;

  if(query_pos.getX()>10.){ // use NN evaluation for attack
    if(state_valid_at != WSinfo::ws->time){
      AbstractMDP::copy_mdp2astate(state);  // ridi05: copy should be rewritten and use WSinfo instead of mdpstate
      //      LOG_POL(0,"Copy mdpstate in this cycle");
      state_valid_at = WSinfo::ws->time;
    }
    Vector orig_ballpos = state.ball.pos;
    state.ball.pos = query_pos;
    //    display_astate(state);
    // move interceptor:
    int interceptor_idx;
    Vector orig_playerpos;

    interceptor_idx = get_interceptor_in_astate(state);
    if(interceptor_idx >=0){
      orig_playerpos = state.my_team[interceptor_idx].pos;
      state.my_team[interceptor_idx].pos = state.ball.pos + (Vector(-.3,0));
    }
    // display_astate();
    evaluation = Planning::Jnn(state);
    // move interceptor back:
    if(interceptor_idx >=0){
      state.my_team[interceptor_idx].pos = orig_playerpos;
    }
    state.ball.pos = orig_ballpos;

    // transform evaluation to the range of -50 to 90
    //    evaluation = 90. - (90.-(-50.)) * evaluation;
    evaluation = 10. + 90. *(1.0 -evaluation);
    //    LOG_POL(0,"Neural Evaluation of query pos  "<<query_pos<<": "<<evaluation);
  }
  else{  // standard evaluation
    evaluation = evaluate_pos_analytically(query_pos);
    // LOG_POL(0,"Standard evaluation of  "<<query_pos<<": "<<evaluation);
  }
    //  return evaluation;
  return evaluation;
}


double Tools::evaluate_potential_of_pos(const Vector pos){
  Vector query_pos;
  
  query_pos = check_potential_pos(pos);
  
  double evaluation = evaluate_pos(query_pos);

  LOG_POL(0,"Evaluation of pos "<<pos<<" : potential pos "<<query_pos<<" evaluation "<<evaluation);
  return evaluation;
}


int Tools::compare_two_positions(Vector pos1, Vector pos2){
  if(evaluate_potential_of_pos(pos1) >= evaluate_potential_of_pos(pos2))
    return 1;
  return 2;
}

void Tools::display_direction(const Vector pos, const ANGLE dir, const double length, const int color){
#if LOGGING && BASIC_LOGGING
  const char* COLORS[]={"000000","FFFFFF","AAAAAA","0000FF","00FF00","FF0000"};
#endif

  Vector targetpos;
  targetpos.init_polar(length, dir);
  targetpos+=pos;
  LOG_POL(0,_2D<<VL2D(pos, targetpos, COLORS[color]));
}

// predicates ued for evaluation

bool Tools::can_advance_behind_offsideline(const Vector pos){

  if (pos.getX() < 0.0)
    return false;

  if(pos.getX() > WSinfo::his_team_pos_of_offside_line())
    return true;  // already behind offside line

  if(WSinfo::his_team_pos_of_offside_line()>FIELD_BORDER_X-10. && fabs(pos.getY()) >15.)
    return false;  // offside already ok.

  // new 05: check, if open area in front

  Vector endofregion;
  double length = 100.;
  double width = 125.; //should be >= length
  endofregion.init_polar(length, 0);
  endofregion += pos;
  Quadrangle2d check_area = Quadrangle2d(pos, endofregion, 6.,width);
  //  LOG_POL(0,<<_2D<< check_area );
  PlayerSet pset = WSinfo::valid_opponents;
  pset.keep_players_in(check_area);

  if(pset.num == 0){
    return true;
  }

  return pset.num == 1 && pset[0] == WSinfo::his_goalie;
}


int Tools::potential_to_score(Vector pos){
  
  if(can_score(pos))
     return VERY_HIGH;
  
  if(pos.getX() > FIELD_BORDER_X -  8 && fabs(pos.getY()) < 8)
    return HIGH;
  if(pos.getX() > FIELD_BORDER_X - 12 && fabs(pos.getY()) <12)
    return MEDIUM;
  if(pos.getX() > FIELD_BORDER_X - 16 && fabs(pos.getY()) <18)
    return LOW;
  return NONE;
}

bool Tools::can_actually_score(const Vector pos){
  return can_score(pos, true); // consider goalie when testing
}


bool Tools::can_score(const Vector pos, const bool consider_goalie){

  Vector opgoalpos= Vector(FIELD_BORDER_X, 0.);

  if (pos.distance(opgoalpos) > 20.)
    return false;

  PlayerSet pset;
  Vector endofregion;
  double width;
  // check left corner
  Vector testcorner =Vector(FIELD_BORDER_X, 7.);
  endofregion.init_polar(pos.distance(testcorner) *1.4+2,(testcorner -pos).ARG());
  endofregion += pos;
  width = 2* 0.5 * pos.distance(endofregion);
  Quadrangle2d check_area   = Quadrangle2d(pos, endofregion, 1.,width);
  //  LOG_POL(0,<<_2D<< check_area );
  pset = WSinfo::valid_opponents;
  pset.keep_players_in(check_area);
  if(pset.num == 0){
    return true;
  }
  if(pset.num == 1 && pset[0] != WSinfo::his_goalie && consider_goalie == false){
    return true;
  }
  // check right corner
  testcorner =Vector(FIELD_BORDER_X, -7.);
  endofregion.init_polar(pos.distance(testcorner) *1.4+2,(testcorner -pos).ARG());
  endofregion += pos;
  width = 2* 0.5 * pos.distance(endofregion);
  check_area = Quadrangle2d(pos, endofregion, 1.,width);
  //LOG_POL(0,<<_2D<< check_area );
  pset = WSinfo::valid_opponents;
  pset.keep_players_in(check_area);
  if(pset.num == 0){
    return true;
  }
  if(pset.num == 1 && pset[0] != WSinfo::his_goalie && consider_goalie == false){
    return true;
  }
  // check middle
  testcorner =Vector(FIELD_BORDER_X, 0);
  endofregion.init_polar(pos.distance(testcorner) *1.4+2,(testcorner -pos).ARG());
  endofregion += pos;
  width = 2* 0.5 * pos.distance(endofregion);
  check_area = Quadrangle2d(pos, endofregion, 1.,width);
  //LOG_POL(0,<<_2D<< check_area );
  pset = WSinfo::valid_opponents;
  pset.keep_players_in(check_area);
  if(pset.num == 0){
    return true;
  }
  if(pset.num == 1 && pset[0] != WSinfo::his_goalie && consider_goalie == false){
    return true;
  }
  return false;
}

bool Tools::opp_can_score(const Vector pos)
{

  Vector mygoalpos= Vector(-FIELD_BORDER_X, 0.);

  if (pos.distance(mygoalpos) > 22.)
    return false;

  PlayerSet pset;
  Vector endofregion;
  double width;
  // check left corner
  Vector testcorner =Vector(-FIELD_BORDER_X, 7.);
  endofregion.init_polar(pos.distance(testcorner) * 1.4 + 2,(testcorner - pos).ARG());
  endofregion += pos;
  width = 2* 0.33 * pos.distance(endofregion);//TG09:0.5->0.33
  Quadrangle2d check_area   = Quadrangle2d(pos, endofregion, 1.,width);
    LOG_POL(0,<<_2D<< check_area );
  pset = WSinfo::valid_teammates;
  pset.keep_players_in(check_area);
  if (pset.num == 0)
    return true;
 
  // check right corner
  testcorner = Vector(-FIELD_BORDER_X, -7.);
  endofregion.init_polar(pos.distance(testcorner) *1.4+2,(testcorner -pos).ARG());
  endofregion += pos;
  width = 2* 0.33 * pos.distance(endofregion);//TG09:0.5->0.33
  check_area = Quadrangle2d(pos, endofregion, 1.,width);
  LOG_POL(0,<<_2D<< check_area );
  pset = WSinfo::valid_teammates;
  pset.keep_players_in(check_area);
  if(pset.num == 0)
    return true;

  // check middle
  testcorner =Vector(-FIELD_BORDER_X, 0);
  endofregion.init_polar(pos.distance(testcorner) *1.4+2,(testcorner -pos).ARG());
  endofregion += pos;
  width = 2* 0.33 * pos.distance(endofregion);//TG09:0.5->0.33
  check_area = Quadrangle2d(pos, endofregion, 1.,width);
  LOG_POL(0,<<_2D<< check_area );
  pset = WSinfo::valid_teammates;
  pset.keep_players_in(check_area);

  return pset.num == 0;
}


bool Tools::close2_goalline(const Vector pos){
  return (pos.getX() > FIELD_BORDER_X - 7);
}

int Tools::compare_positions(const Vector pos1, const Vector pos2, double & difference){
  difference = 0;
#if 0 // ridi06. 13.6. make this much more restrictiv. only prefer actual scorers!
  if(potential_to_score(pos1)>potential_to_score(pos2))
    return FIRST;
  if(potential_to_score(pos1)<potential_to_score(pos2))
    return SECOND;
  
  if(potential_to_score(pos1)>= MEDIUM){ // ridi 06
    // the potential to score is the same for both positions.
    // ridi06: restrict this to scoring positions. otherwise close2goalline positions are prefered.
    // checking the ability to overcome offside is asked below
    if(close2_goalline(pos1) == true && close2_goalline(pos2) == false)
      return FIRST;
    if(close2_goalline(pos1) == false && close2_goalline(pos2) == true)
      return SECOND;
    if(close2_goalline(pos1) == true && close2_goalline(pos2) == true){
      // they're both very far advanced, so take their y -coordinate 
      //    LOG_POL(0,"pos 1 : "<<pos1<<" are close to borderlien pos2 "<<pos2);
      difference =  (FIELD_BORDER_Y - fabs(pos1.y))  - (FIELD_BORDER_Y - fabs(pos2.y));
      return EQUAL;
    }
  }
#endif

  if(can_actually_score(pos1)== true &&can_actually_score(pos2)== false)
    return FIRST;
  if(can_actually_score(pos1)== false && can_actually_score(pos2)== true)
    return SECOND;


  if(WSinfo::his_team_pos_of_offside_line() > FIELD_BORDER_X - 12.){ // already pretty advanced
    // try to prefer positions closer to the middle
    if(pos1.getX() > FIELD_BORDER_X - 16. && fabs(pos1.getY())+5.< fabs(pos2.getY())) // do not pass too far back, improve y
      return FIRST;
    if(pos2.getX() > FIELD_BORDER_X - 16. && fabs(pos2.getY())+5.< fabs(pos1.getY())) // do not pass too far back, improve y
      return SECOND;
  }
  // the potential to score is the same for both positions. (probably None)
  // none of the positions is very close2 goalline
  // now, compare their ability to overcome offside
  if(WSinfo::his_team_pos_of_offside_line()< FIELD_BORDER_X -12){ // ridi06: the offside line still might be improved, so test for this.
    if(pos1.getX() > WSinfo::his_team_pos_of_offside_line() && pos2.getX() < WSinfo::his_team_pos_of_offside_line())
      return FIRST;
    if(pos1.getX() < WSinfo::his_team_pos_of_offside_line() && pos2.getX() > WSinfo::his_team_pos_of_offside_line())
      return SECOND;
    
    if(can_advance_behind_offsideline(pos1) == true && can_advance_behind_offsideline(pos2) == false){
      //LOG_POL(0,"pos 1 : "<<pos1<<" can advance behind offside , but 1 cannot "<<pos2);
      return FIRST;
    }
    if(can_advance_behind_offsideline(pos1) == false && can_advance_behind_offsideline(pos2) == true){
      //LOG_POL(0,"pos 2 : "<<pos2<<" can advance behind offside , but 1 cannot "<<pos1);
      return SECOND;
    }
  }  // the offside line is behind the 12m line


  // both positions can either overcome offside or both fail. 
  // now, check their principal evaluation.
  //  LOG_POL(0,"Equal predicates. Now evaluate: pos 1 : "<<pos1<<"  pos2 "<<pos2);
  difference = evaluate_pos(pos1) - evaluate_pos(pos2);
  return EQUAL;
}


bool Tools::is_pos1_better(const Vector pos1, const Vector pos2){
  double evaluation_delta;
  if (compare_positions(pos1, pos2, evaluation_delta) == FIRST)
    return true;
  else if (compare_positions(pos1, pos2, evaluation_delta) == SECOND)
    return false;
  else if(evaluation_delta >0)
    return true;
  else
    return false;
  return false;
}


bool Tools::is_pass_behind_receiver(const Vector mypos, const Vector receiverpos, const double passdir){

  ANGLE line2player = (receiverpos - mypos).ARG();
  ANGLE line2goal = (Vector(52.0,0) - mypos).ARG();
  
  bool is_behind = true;

  /* todo: not so easy. Kann Psse verhindern, die nach vorne gespielt werden. Muss verfeinert werden, sonst
     viel zu restriktiv. */
  
  return false;


  
  if(line2goal.diff(ANGLE(passdir)) < line2goal.diff(line2player))
    is_behind = false;

  if(is_behind){
    display_direction(mypos, ANGLE(passdir), 20.);
    DBLOG_DRAW(0,VC2D(receiverpos,2.0,"brown"));
    DBLOG_POL(0,"Tools: Winkel. zw. Mir/Tor und Spielerpos: "<<RAD2DEG(line2goal.diff(line2player))
	      <<" ist grer als zw. Mir/Tor und passdir: "<<RAD2DEG(line2goal.diff(ANGLE(passdir))));
    DBLOG_POL(0,"Tools: Is pass behind receiver: YES. (receiverpos is brown)");
    return true;
  }
  return false;
}

bool
Tools::willMyStaminaCapacitySufficeForThisHalftime()
{
  if (WSinfo::ws->time > 2*10*ServerOptions::half_time) return true; //ignore extra time
  int halfTime = (WSinfo::ws->time < 10*ServerOptions::half_time) ? 1 : 2;
  
  float usedStaminaSoFar = (float)(  ServerOptions::stamina_capacity + ServerOptions::stamina_max
                                   - (WSinfo::me->stamina_capacity + WSinfo::me->stamina) );
  float staminaPerTimeStep 
    = (halfTime==1) ? usedStaminaSoFar / (float)WSinfo::ws->time
                    : usedStaminaSoFar / (float)(WSinfo::ws->time-10*ServerOptions::half_time);
  float remainingStaminaSufficesForAsManyTimeSteps
    = (WSinfo::me->stamina_capacity + WSinfo::me->stamina) / staminaPerTimeStep;
  int timeStepsTillHalfTime 
    = (halfTime==1) ? 10*ServerOptions::half_time - WSinfo::ws->time
                    : 2*10*ServerOptions::half_time - WSinfo::ws->time;
  
  int timeStepsWithoutStamina 
    = timeStepsTillHalfTime - remainingStaminaSufficesForAsManyTimeSteps;
  if (timeStepsWithoutStamina <= 0)
    return true;
  else
    return false;
}


bool
Tools::willStaminaCapacityOfAllPlayersSufficeForThisHalftime()
{
	for (int i = 0; i < WSinfo::alive_teammates.num; i++ )
	{
		PPlayer curr = WSinfo::alive_teammates[i];
		if ( curr == NULL ) continue;
		if ( curr->stamina_capacity_bound > 20000 ) continue;
  	if (WSinfo::ws->time > 2*10*ServerOptions::half_time) return true; //ignore extra time
  	int halfTime = (WSinfo::ws->time < 10*ServerOptions::half_time) ? 1 : 2;
  
  	float usedStaminaSoFar = (float)(  ServerOptions::stamina_capacity + ServerOptions::stamina_max
                                   - (curr->stamina_capacity_bound) );
    float staminaPerTimeStep 
    	= (halfTime==1) ? usedStaminaSoFar / (float)WSinfo::ws->time
                    : usedStaminaSoFar / (float)(WSinfo::ws->time-10*ServerOptions::half_time);
  	float remainingStaminaSufficesForAsManyTimeSteps
    	= (curr->stamina_capacity_bound) / staminaPerTimeStep;
  	int timeStepsTillHalfTime 
    	= (halfTime==1) ? 10*ServerOptions::half_time - WSinfo::ws->time
      	              : 2*10*ServerOptions::half_time - WSinfo::ws->time;
  
  	int timeStepsWithoutStamina 
    	= timeStepsTillHalfTime - remainingStaminaSufficesForAsManyTimeSteps;
  	if (timeStepsWithoutStamina > 0)
  	{
  		LOG_POL(1,  << "willStaminaCapacityOfAllPlayersSufficeForThisHalftime == false because of player " << curr->number 
  								<< " with stamina_capacity_bound " << curr->stamina_capacity_bound);
    	return false;
    }
  }
  return true;
}


ANGLE Tools::degreeInRadian (double degree)
{
    return ANGLE (0.017453293 * degree);
}


double Tools::radianInDegree (ANGLE radian)
{
    return radian.get_value_0_p2PI () * (180.0 / PI);
}


void
Tools::setModelledPlayer (PPlayer p)
{
    modelled_player = p; //TG16
}
