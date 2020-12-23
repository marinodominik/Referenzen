// vim:fdm=marker:ts=2:sw=2
#ifndef _STANDARD_SITUATION_BMP_H_
#define _STANDARD_SITUATION_BMP_H_

// #include "noball05_bmp.h"
#include "../basics/wmdef.h"
#include "skills/neuro_go2pos_bms.h"
#include "skills/neuro_kick05_bms.h"
#include "skills/neuro_intercept_bms.h"
#include "skills/basic_cmd_bms.h"
#include "wball06_bmp.h"
#include "skills/face_ball_bms.h"
#include "skills/score05_sequence_bms.h"
#include "play_pass_bms.h"
#include "../basics/intention.h"

// "Wildcard"-IDs for all situations where we or opponent has the ball
// WTH -- warum zusaetzliche konstanten!???
const int PMS_my_Kick  = 30;
const int PMS_his_Kick  = 31;

/**
 * @class StandardSituation
 * Complete rewrite (3/2006) by hasan.
 */
class StandardSituation: public BodyBehavior
{
	private:  // data {{{1
		static bool svIsInitialized;
		StandardSituation();
		static StandardSituation* svpInstance;

		/// my back number
		int ivMyNumber;
		
		/// ball position (may differ from WSinfo bc of additional playmode info)
		Vector ivBallPos;

		/// number of kicking player (if it is our turn), negative otherwise
		int ivKickerNr;

		/// whether we kick
		bool ivWeShallKick;

		// Other behaviors {{{
		BasicCmd*         ivpBasicCmd;
		Score05_Sequence* ivpScore;
		NeuroIntercept*   ivpIntercept;
		NeuroGo2Pos*      ivpGo2Pos;
		Wball06*          ivpWBall;
		FaceBall*         ivpFaceBall;
		NeuroKick05*      ivpNeuroKick;
		PlayPass*         ivpPlayPass;
		// }}}

		//  Timing Variables {{{
		/// last time this behavior was active
		int ivLastActivationTime;

		/// start time of active StandardSituation
		int ivActSitStartTime;

		/// duration of active StandardSituation
		int ivActSitDuration;
		
		/// playmode when last active
		PlayMode ivLastPlayMode;

		/// time left to kick
		int ivTimeLeft;
		// }}}

		// Own actions {{{
		/// Number of kicks in a row
		int ivConsecKicks;
		static int svMaxConsecKicks;
		
		/// last executed behavior
		BodyBehavior* ivpLastBehavior;

		/// speed of last kick
		double ivLastKickSpeed;
		
		/// target of last kick
		Vector ivLastKickTarget;
		
		/// last intention
		Intention ivLastIntention;

		// }}}
		
		// Stuff from previous version... {{{
		static Vector svKickoffTarget;
		static double svKickoffSpeed;
		static int    svPassAnnounceTime;
		static int max_ball_age_when_kick;
		static double homepos_tolerance;
		static int homepos_stamina_min;
		static double clearance_radius_min;
		// }}}
		
		
	public: // {{{1
		/** @return true  if StandardSituation set command
		 *  @return false otherwise */
		bool get_cmd(Cmd & cmd);

		/**
		 * initialize static variables of StandardSituation and dependants
		 */
		static bool init(char const * conf_file, int argc, char const* const* argv);

		/** 
		 * Returns instance of singleton. {{{ 
		 */
		static StandardSituation* getInstance(){
			if(!svpInstance)
				svpInstance = new StandardSituation;
			return svpInstance;
		}
		// }}}

		virtual ~StandardSituation(){ // {{{
			delete ivpScore;
			delete ivpIntercept;
			delete ivpBasicCmd;
			delete ivpFaceBall;
			delete ivpGo2Pos;
			delete ivpWBall;
			delete ivpNeuroKick;
			delete ivpPlayPass;
		}
		//}}}

	private: // Functions {{{1

		/** tries to continue last command, if existing */
		bool getCmdContinued(Cmd&);

		/** write current intention to blackboard */
		void write2blackboard();

		/** sorts players to select kicker */
		double getKickerVal(const PPlayer&);
		
		/** @return back number of player who shall kick the ball, 
		 *  @return -1 if not applicable */
		int getKickerNumber();
		
		/** @return Ball pos (has some intelligence regarding playmode) */
		Vector getCleanBallPos();

		/** @return direction kicker has to turn towards */
		Vector getKickerTurnTarget();

		/** Get cmd for kicking player */
		bool getCmdKicker(Cmd&);

		/** Get cmd for player who is not kicking */
		bool getCmdNonKicker(Cmd&);

		/** @return true if I intercept  */
		bool getCmdReactOnPassInfo(Cmd&, PPlayer,const Vector&);

		/** @return true if I replace missing start player */
		bool getCmdReplaceMissingStartPlayer(Cmd&);

		/** Get cmd for a kick into the unknown */
		bool getCmdPanicKick(Cmd&);

		// Walk-to Targets
		/** @return target for nonstart-player */
		Vector getTargetNonStartPlayer();
		/** @return target for nonstart-player in myCornerKick playmode */
		Vector getTargetNonStartPlayerMyCorner();
		/** @return target for nonstart-player in myFreeKick playmode */
		Vector getTargetNonStartPlayerMyFreeKick();
		/** @return target for nonstart-player in myKickIn playmode */
		Vector getTargetNonStartPlayerMyKickIn();

		/** @return true if the pass_info from player is "the signal" */
		bool isPassInfoTheSignal(PPlayer);

		enum CornerRole{
			L1_L, L1_M, L1_R,
			L2_L, L2_M, L2_R,
			NOT_INVOLVED
		};

		/** @return my role in a corner kick */
		CornerRole getCornerRole();

		/** modify dash for better stamina handling */
		void modifyDashCmd(Cmd&);

		/** set a command that turns player every cycle */
		bool getCmdScanField(Cmd&);

		/** set a command that turns player every 2nd cycle */
		bool getCmdScanField2nd(Cmd&);

		/** Say whether we shall kick or not */
		bool shallWeKick(int);

		/** use go2pos, avoid 9m-clearance zone, consider stamina */
		bool getCmdGo2PosWithCare(Cmd&,Vector,bool,double dist=0);

		/** returns whether no teammate is near ball */
		bool isStartPlayerMissing();

		/** @return cycles to kick */
		int getTimeLeft();

		/** @return tangent intersection point of circle */
		Vector getTangentIsct(const Vector& cM, double cR, const Vector& mypos, const Vector& dest);

		/** get pass from wball */
		bool getPassIntention(Intention&);
};


#endif
