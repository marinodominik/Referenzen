#ifndef _DRIBBLE_BETWEEN06_H
#define _DRIBBLE_BETWEEN06_H

// includes
#include "../../basics/Cmd.h"
#include "dribble_around06.h"
#include "dribble_between_actions.h"

#define MIN_STAMINA_DRIBBLE  ((0.05+ServerOptions::effort_dec_thr)*ServerOptions::stamina_max)
                             //TG09: alt: 1600

	

/**
 * DribbleBetween06
 *
 * Uses DribbleAround06 to find a way between opponents on the way to goal
 * @author Hannes Schulz <mail@hannes-schulz.de>
 * @version 0.1
 *
 */

class DribbleBetween06:public BodyBehavior
{
	private:
		static DribbleBetween06* myInstance;              ///< Singleton
		DribbleBetween06();                               ///< Private constructor
		DribbleBetween06(const DribbleBetween06&);          ///< Copying also prohibited!
		DribbleBetween06 operator=(const DribbleBetween06&);///< Copying still prohibited!

		/// the DribbleAround06 behavior we're using
		DribbleAround06* dribbleAround;

		Cmd cachedCmd;
		int cachedCmdTime;


		enum Mode{
			DB_NO_OPP,
			DB_ONE_OPP,
			DB_TWO_OPP,
			DB_TOO_MANY_OPP};
		Mode mode;

		/// get the relevant opponents
		void setRelevantOpponents();

		/// select point to dribble to
		Vector getTargetPos();
		Vector getTargetPosOld();

		/// if I'm attacked go straight, do not turn
		bool iAmAttacked();

		/// judge how good a dribble dir is
		/// @param dribbleAngle relative to body
		float getValueForDribbleDir(float dribbleAngle);

		/// target to dribble to
		Vector dribbleTo;
		/// dribbleAround target
		Vector dribbleAroundTarget;
		float dribbleAroundTargetValue;
		bool keepBall;
		int maxSpeed;
		bool chooseDirMyself;

		/// target last time
		Vector lastTarget;

		/// the opponents we're playing against
		PlayerSet opps;

		/// safes result of is_dribble_between_possible
		bool isDribblingSafe;
		bool dribblingInsecure;

		/// last time when we dribbled
		int lastDribbleTime;

		/// how often we dribblet to current target
		int targetTimeCounter;

		/// The statistics are saved in here
#define DRIBBLE_STATS_BUFFER_LENGTH 10
		class Stats {
			private:
				struct DribbleState {
					PPlayer closestOpp;
					float xDistToClosestOpp;
					DribbleAction actionTaken;
					bool actionSucceeded;
					int time;
					bool didDribble;
				};
				// The ring buffer of dribbleStates
				DribbleState dribbleStates[DRIBBLE_STATS_BUFFER_LENGTH];
				// The current Position in the ring buffer
				int bufferPos;

				DribbleBetween06* db;

			public:
				// update from viewpoint of is_dribble_between_possible
				void updateTestDribbling();

				// update from viewpoint of get_cmd
				void updateGetCmd();

				// get some statistics output
				void getStatistics();

				// constructor
				Stats();

		} stats;

	public:
		/// get the only instance of this class (Singleton pattern!)
		static DribbleBetween06* getInstance();

		/// Try to dribble.
		/// @return true   if successful
		/// @return false  otherwise
		bool get_cmd(Cmd& cmd);

		inline void set_choose_dir_yourself(bool b){
		  chooseDirMyself = b;
		}
		inline bool get_choose_dir_yourself(){
		  return chooseDirMyself;
		}
		inline void set_max_speed(int i){
		  maxSpeed = i;
		}
		inline int get_max_speed(){
		  return maxSpeed;
		}
		/// Set a point to dribble to
		void set_target(const Vector&);

		/// Set whether to keep ball in kickrange
		inline void set_keep_ball(bool b){
		  keepBall = b;
		}
		inline bool get_keep_ball(){
		  return keepBall;
		}
		
		/// Get the point to dribble to
		Vector get_target();

		/// say whether dribbling is generally possible
		bool is_dribble_safe(bool = true);
		bool is_dribble_insecure();

		/// say whether neck request was set
		bool is_neck_req_set();
		ANGLE get_neck_req();

		virtual ~DribbleBetween06();

		static bool init(char const * conf_file, int argc, char const* const* argv) {
			cout<<"\n DribbleBetween06: init"<<endl;
			return (DribbleAround06::init(conf_file,argc,argv));
		}
};

#endif /* _DRIBBLE_BETWEEN06_H */
