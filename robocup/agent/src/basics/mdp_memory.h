/* @ralf:
   This is a class to store a small history, like which player was the last at ball etc..
   MDPstate cannot give such information because it is a look at the current situation.
*/
#ifndef _BS99_MEMORY_
#define _BS99_MEMORY_

class MDPmemory {
 protected:
  static const int MAX_MOMENTUM=5;
  int momentum[MAX_MOMENTUM];
  int counter;

 public:
  MDPmemory();
  /** This function updats the history information. It must be called every cyrcle to be correct.
   */
  void update();

  int opponent_last_at_ball_number;
  int opponent_last_at_ball_time;
  int teammate_last_at_ball_number; // include myself!
  int teammate_last_at_ball_time;
  int team_last_at_ball;
  int team_in_attack; //gibt an, welches Team im Angriff ist
};

#endif
