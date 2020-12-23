#ifndef _DRIBBLE_AROUND06_H
#define _DRIBBLE_AROUND06_H

// includes
#include <vector>

#include "../../basics/Cmd.h"
#include "dribble_around.h"
#include "dribble_straight_bms.h"
#include "neuro_go2pos_bms.h"
#include "basic_cmd_bms.h"
#include "search_ball_bms.h"
#include "one_step_kick_bms.h"
#include "intercept_ball_bms.h"
#include "dribble_between_actions.h"
#include "onetwo_holdturn_bms.h"

/**
 * 
 * Dribble Around
 * a new implementation of dribbling that explicitly keeps the ball
 * away from the next player around.
 * @author Hannes Schulz <mail@hannes-schulz.de>
 * @version 0.9
 * 
 */

class DribbleAround06:public BodyBehavior
{
	public:
		enum DARequest{
			DAREQ_NONE,
			DAREQ_DASH,
			DAREQ_KICK,
			DAREQ_TURN
		};
	private:
		// behaviors that we're using
		BasicCmd* basic_cmd;
		DribbleStraight* dribble_straight;
		NeuroGo2Pos* go2pos;
		OneStepKick* onestepkick;
		SearchBall* searchball; ///< only for testing! (TODO)
		InterceptBall* intercept;
		OneTwoHoldTurn* holdturn;

		static DribbleAround06* myInstance;             ///< Singleton
		DribbleAround06();                              ///< Private constructor
		DribbleAround06(const DribbleAround06&);          ///< Copying also prohibited!
		DribbleAround06 operator=(const DribbleAround06&);///< Copying still prohibited!

		/// The destination for the dribble procedure
		Vector dribbleTo;
		/// The player we're dribbling against
		PPlayer opp;
		/// Whether to keep the ball in kickrange
		bool keepBallSafe;
		/// The maximum speed
		int maxSpeed;

		DARequest request;

		int requestTime;

		// set this if cmd was actually executed
		bool didDribble;

		// is dribbling insecure
		bool dribbleInsecure;

		/// return a set or "interesting" opponents
		PPlayer getRelevantOpponent();

		/// generate a command that kicks to a position
		/// @param pos where ball has to be in next cycle
		/// @param keepInKickRadius whether to keep ball
		/// @param force whether to ignore closeness errors
		bool getCmdKickToDest(Cmd&,const Vector&,bool,bool);

	public:
		/// Saves a balls state, f.ex. of next cycle
		struct BallState {
			Vector pos;
			Vector vel;
			void setAssumeNoAction(const Ball*);
			void setNowTestCollision(const PPlayer&,const Ball*);
			void setNow(const Ball*);
			void setNext(const BallState&);
		};

		/// Saves a players state, f.ex. of next cycle
		struct PlayerState {
			PPlayer origPlayer;
			Vector pos;
			Vector vel;
			int  age;
			float kick_radius;
			bool reachesPos; // whether kick_radius overlaps with assumed target pos
			ANGLE ang;
			bool movedToAvoidCollision;
			void setAssumeNoAction(const PlayerState&,const Vector&);   // ball/player pos
			void setAssumeNoAction(const PPlayer,const Vector&);   // ball/player pos
			void setAssumeNoAction(const PPlayer,const BallState&);// ball pos
			void setAssumeToPos(const PPlayer,const Vector&);      // ball/player pos
			void setAssumeToPos(const PPlayer,const Ball*);    // ball pos
			void setAssumeToPos(const PPlayer,const BallState&);   // ball pos
			void setAssumeToPos(const PPlayer,const PlayerState&); // player pos
			void setNowTestCollision(const PPlayer,const Ball*);            //
			void setNow(const PPlayer,const Ball*);            //
			void testCollision(const PlayerState&);  // moves player if collision anticipated
		};

	private:
		PlayerState thisMe, nextMeNA, nextOppNA, nextOppToBall, nextOppToMe;
		BallState thisBall, nextBall;

		/// define possible actions to take
		enum Action { DA_KICK, DA_DASH, DA_TURN, DA_COLKICK, DA_TACKLE, DA_GOALK };

		/// return the next action to take
		Action getNextAction(PPlayer&);
		/// return the next action (no opponent)
		Action getNextAction();

		/// find positions on outer rim of kick_radius
		void getTargetsInMe(Vector& safestPos, Vector& bestPos);

		/// get the destination to kick to when ball is supposed
		/// to be at pos in n cycles. tooFar is set if the dest 
		/// cannot possibly be reached.
		Vector getKickDestForBallPosInNCycles(const Vector&, const int, bool&);

		/// returns the players distance to the way of the ball
		/// assuming ball moves from now on with speed vel
		/// steps becomes number of cycles until player can intercept ball
		float getPlayerDistToBallTraj(const PPlayer&, const Vector&, int&);

		// special getCmds
		bool getColKickCmd(Cmd&);
		bool getTackleCmd(Cmd&);
		bool getTurnCmd(Cmd&);
		bool getDashCmd(Cmd&);
		bool getKickCmd(Cmd&);
		bool getKickAhead(Cmd&);
		bool getKickAheadBallOK(Cmd&,ANGLE,bool);
		bool getKickAheadPrepareBall(Cmd&, bool, bool);
		bool getKickForTurn(Cmd&);

		bool getGoalieKickCmd(Cmd&);

	public:
		void setRequest(DARequest);

		inline DARequest getRequest(){
			return request;
		}
		/// Use this function to get an instance of DribbleAround06
		static DribbleAround06* getInstance();

		/// set the position to dribble to
		void set_target(const Vector&);

		/// say whether to keep the ball in the own kick range
		/// (maybe because there is another enemy ahead)
		void set_keepBall(bool);

		/// Set the maximum speed
		void set_max_speed(int);

		/// Try to dribble.
		/// @return true   if successful
		/// @return false  otherwise
		bool get_cmd(Cmd& cmd);
		bool get_dribble_straight_cmd(Cmd& cmd);

		// set this if the command was actually executed
		void setDribbled(bool);


		// check whether dribbling is insecure
		bool isDribbleInsecure();

		// reset request, used when dribbleDir changes
		void resetRequest();

		// a value for statistics
		DribbleAction lastActionTaken;

		// neck requests
		bool neckReqSet;
		ANGLE neckReq;

		virtual ~DribbleAround06();

		static bool init(char const * conf_file, int argc, char const* const* argv) {
			return (
					NeuroGo2Pos::init(conf_file,argc,argv)
					&& BasicCmd::init(conf_file,argc,argv)
					&& DribbleStraight::init(conf_file,argc,argv)
					&& SearchBall::init(conf_file,argc,argv)
					&& OneStepKick::init(conf_file,argc,argv)
					&& OneTwoHoldTurn::init(conf_file,argc,argv)
					&& InterceptBall::init(conf_file,argc,argv));
		}

		void getNumTurnsAndPosAfter(ANGLE,int&,Vector,Vector,ANGLE,Vector&,Vector&,ANGLE&,Vector*,const double&);
		void fillDashArray(Vector,Vector,ANGLE,Vector*,int,double);
		vector<Vector> getBestDestOnMeInNSteps(const int&,const Vector&,const ANGLE&,const bool&,const bool&);
		bool amIFastestToPos(const Vector&,int);
		bool getKickForTurnAndDash(Cmd&);
		bool iCanGetBallByDashing();
};


#endif  /* _DRIBBLE_AROUND06_H */


