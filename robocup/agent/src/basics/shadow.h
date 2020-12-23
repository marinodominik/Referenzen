#ifndef _SHADOW_H_
#define _SHADOW_H_

#include "globaldef.h"

/*
 *
 * This class calculates the area on the goal, which is covered by the goalie,
 * assuming he is on a point specified by the user.
 * The position of the ball must be given by the user, too.
 *
 * */

class Shadow
{
private:
	int prediction_depth;	//Angabe, wie weit die Vorhersagen gehen sollen
	Vector point;				//Punkt, von dem aus Schatten geworfen wird
	Vector goalie;				//Position des Goalies
	Triangle2d schatten;		//Dreieck vom Ball zu den Torpfosten. Wird berechnet, sobald "point" gesetzt ist.
	double cover; 				//Vom Goalie abgedeckte Fläche des Schattens in Prozent
	Vector left_open_point;		//Erster linker Punkt vor dem Tor, der nicht mehr abgedeckt wird
	Vector right_open_point;	//Erster rechter Punkt vor dem Tor, der nicht mehr abgedeckt wird
	Vector GOAL_LEFT_CORNER;      
	Vector GOAL_RIGHT_CORNER;
	Vector GOAL_CENTER; 
	/*
	 * 							Links
		Spieler -> O-------------->		
	 *							Rechts
	 */	
	
public:
	Shadow();	//Setzt "prediction_depth" auf 1
	~Shadow();
	//Setter
	void set_point(const Vector point);	//(!) Muss gesetzt werden
	void set_goalie(const Vector goalie);
	void set_goal(const int goal); // 0=His Goal(defualt) , 1=My Goal
	void set_prediction_depth(const int depth);
	//Getter
	double get_cover();
	double get_cover_center();//if near center bonus percent
	Vector get_left_open_point();
	Vector get_right_open_point();
	void get_open_points(Vector &left, Vector &right);
	double get_left_open_area();	//Größe der linken Seite des Tors, die getroffen werden kann
	double get_right_open_area();	//Größe der rechte Seite des Tors, die getroffen werden kann
	void get_open_areas(double &left, double &right);
	double get_goalie_intercept_distance(int steps);
};

#endif
