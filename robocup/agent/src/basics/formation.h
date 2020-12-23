#ifndef _FORMATION_H_
#define _FORMATION_H_

#include "globaldef.h"
#include "PlayerSet.h"
#include "ws_info.h"

class BaseFormation {
 public:
  virtual int get_role(int number)= 0;
 
  virtual Vector get_grid_pos(int number)= 0;
  virtual bool   need_fine_positioning(int number)= 0;
  virtual Vector get_fine_pos(int number)= 0;

  virtual void get_boundary(double & defence, double & offence)= 0;
};

class Formation433 : BaseFormation {
  struct Home {
    Vector pos;
    double stretch_pos_x;
    double stretch_neg_x;
    double stretch_y;
    int role;
  };
  Home home[NUM_PLAYERS+1];
  int boundary_update_cycle;
  double defence_line, offence_line; Vector intercept_ball_pos; //this values are set in get_boundary, and can be used as cached values in the same cycle
 public:
  //Formation433() { init(0,0,0); }// test
  bool init(char const * conf_file, int argc, char const* const* argv);
  int get_role(int number);
  Vector get_grid_pos(int number);
  bool   need_fine_positioning(int number);
  Vector get_fine_pos(int number);

  void get_boundary(double & defence, double & offence);
  double defence_line_ball_offset;
};

//////////////////////////////////////////////////////////////////////////////
// CLASS Formation05 by tga
//////////////////////////////////////////////////////////////////////////////
/**
 * Class Formation05
 */
class Formation05:BaseFormation
{
  ////////////////////////////////////////////////////////////////////////////
  // PROTECTED AREA
  ////////////////////////////////////////////////////////////////////////////
  protected:
    struct HomePosition
    {
      Vector pos;
      double stretch_pos_x;
      double stretch_neg_x;
      double stretch_y;
      int role;
    };
    struct DirectOpponentAssignment
    {
      int directOpponent;
      int previousDirectOpponent;
    };
    struct DistanceCollection
    {
      PPlayer myPlayer;
      PPlayer hisPlayer;
      float   distance;
      DistanceCollection();
    };
    struct CoachAssignmentUpdateInformation
    {
      long time[2];
      int  numberOfAssignments[2];
      CoachAssignmentUpdateInformation();
      void update();
    };
    HomePosition              ivHomePositions[NUM_PLAYERS+1];
    static int                cvMyCurrentDirectOpponentAssignment;
    DirectOpponentAssignment  ivDirectOpponentAssignments[NUM_PLAYERS+1];
    CoachAssignmentUpdateInformation ivCoachAssignmentUpdateInformation;
    double   getCurrentRearmostPlayerLine(int team);
    double   getCurrentForemostPlayerLine(int team);
    double   getCurrentLeftmostPlayerLine(int team);
    double   getCurrentRightmostPlayerLine(int team);
    XYRectangle2d getCurrentFieldPlayerRectangle(int team);

  ////////////////////////////////////////////////////////////////////////////
  // PUBLIC AREA
  ////////////////////////////////////////////////////////////////////////////
  public:
    bool    init(char const * conf_file, int argc, char const* const* argv);
    int     get_role(int number);
    Vector  get_grid_pos(int number);
    bool    need_fine_positioning(int number);
    Vector  get_fine_pos(int number);
    void    get_boundary(double & defence, double & offence);

  double defence_line_ball_offset;
  
  void    computeDirectOpponents();
  bool    checkDirectOpponentAssignments();
  PPlayer findNearestOpponentTo( Vector myPos,
                                 bool * assignableOpps);
  bool    getDirectOpponent(int number, PPlayer & opponent);
  int     getDirectOpponentNumber(int number);
  bool    getResponsiblePlayerForOpponent(int number, PPlayer & respTeammate);
  bool    getDirectOpponentPosition(int number, Vector & pos);
  Vector  getHomePosition(int number, bool initialHomePosition=false);
  Vector  getHomePosition(int number, XYRectangle2d rect, bool initialHomePosition=false);
  bool    relevantTeammateHasAcceptedTheNewAssignment();
};

#endif
