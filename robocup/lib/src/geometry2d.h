#ifndef _GEOMETRY2D_H_
#define _GEOMETRY2D_H_

#include "Vector.h"
#include "angle.h"
#include <string>

struct Set2d {
  virtual bool inside(Vector const & point) const = 0;
  virtual void draw(std::ostream &outStream) const;
};

struct Circle2d : public Set2d {
  Vector center;
  double radius;
  Circle2d() { radius= 0.0; }
  Circle2d(Vector const & c, double r) { center= c; radius= r; }
  bool inside(Vector const & point) const ;
  void draw(std::ostream &outStream) const;
};

struct XYRectangle2d : public Set2d {
  Vector center;
  double size_x;
  double size_y;
  XYRectangle2d() { size_x= 0; size_y= 0; }
  XYRectangle2d(Vector const & c, double s_x, double s_y) { center= c; size_x= s_x; size_y= s_y; }
  XYRectangle2d(Vector const & p1, Vector const & p2);
  bool inside(Vector const & point) const ;
  void draw(std::ostream &outStream) const;
};

struct Triangle2d : public Set2d {
  Vector p1;
  Vector p2;
  Vector p3;
  Triangle2d() {}
  Triangle2d(Vector const & _p1, Vector const & _p2, Vector const & _p3) { p1= _p1; p2= _p2; p3= _p3; }
  bool inside(Vector const & point) const ;
  void draw(std::ostream &outStream) const;
};

struct Quadrangle2d : public Set2d {
/**
   p1 and p3 must be connected by a diagonal of the quadrangle, or equivalently: 
   the points p1,p2,p3,p4 must follow the circumference of the rectangle

    p2           p1                           p3           p1
      +---------+        		        +---------+  
      |         |     <--- OK		        |         |  <--- NOT OK
      |         | 			        |         |  
      +---------+			        +---------+  
    p3           p4			      p2           p4
       
*/
  Vector p1;
  Vector p2;
  Vector p3;
  Vector p4;
  char color[6];
  Quadrangle2d(){};
  Quadrangle2d(Vector const & _p1, Vector const & _p2, Vector const & _p3, Vector const & _p4) { p1= _p1; p2= _p2; p3= _p3; p4= _p4; }
  /**
     quadrangle is like a rectangle , but vertices are not required to be parallel to
     the x or the y axes, and the angles doesn't need to be right !!!
                                  b
                         --------+
      a         ---------        |
       +--------                 |
       |                         |
       |                         |
    p1 +                         + p2
       |                         |
       |                         |
       +--------                 |
      c         ---------        |
                         --------+
                                  d
 
      the distance between  (a and c) is width1
      the distance between  (b and d) is width2 (and doesn't need to be the same as width1)

      the vectors a-c and b-d are parallel and orthogonal to the vector p2-p1

      but p2-p1 doesn't need to be parallel to the x or the y axes
   */
  Quadrangle2d(Vector const & _p1, Vector const & _p2, double width1, double width2);
  Quadrangle2d(Vector const & _p1, Vector const & _p2, double width);// : Quadrangle2d(_p1,_p2,width,width) {}
  bool inside(Vector const & point) const ;
  void setColor(std::string s);
  void draw(std::ostream &outStream) const;
};

struct Halfplane2d : public Set2d {
  Vector pos;
  Vector normal;
  Halfplane2d():normal(1,0) { }

  /** a halfplane is defined by a point [pos] on it's boundary and by
      the normal vector [normal] pointing towards the interior of the 
      half plane. |normal| = 1 is NOT required!!!
   */
  Halfplane2d(Vector const & p, Vector const & n) { pos= p; normal= n; }

  /** a halfplane is specified be a point [pos] on it's boundary and by an angle. 
      all vectors belonging to the halfplane are then:

      pos + k * vec; with vec.arg() between [ang] and [ang + PI]
   */
  Halfplane2d(Vector const & p, ANGLE const & a);
  bool inside(Vector const & point) const ;
};

struct Tube2d: public Set2d {
  /** a tube is an area between two parallel lines
      and the width is the distance of the lines.
      dir is the line in the middle of the tube, pos is some
      position in the middle of the tube

         -------------------



         -------*-----------  dir
                p


         -------------------

     dir1.norm() is NOT relevant.
   */
  Tube2d():dir(1,0) { sqr_width= 0; }
  Vector pos;
  Vector dir;
  double sqr_width;
  void set_width(double w) { sqr_width= w*w; }
  bool inside(Vector const & point) const ;
};

struct HalfTube2d: public Tube2d {
  /** a tube directed tube

         -------------------
         }       
         }       
         }       
       p *------------------>  dir
         |       
         |        
         |
         -------------------

   */
  bool inside(Vector const & point) const ;
};

struct Cone2d : public Set2d {
  Vector pos;
  Vector dir1;
  Vector dir2;
  Cone2d():dir1(1,0),dir2(1,0) { }
  /** a cone is defined by his focal point [pos] and the two directions,
      so that all vectors belonging to the cone have the form:

      pos + k * vec; with vec.arg() between [dir1.arg()] and [dir2.arg()] (going counterclockwise from ang1 to ang2)
      
             / dir2
            /
           /
          /
       p /
         \
          \
           \
            \
             \ dir1

      dir1.norm() and dir2.norm() are NOT relevant.
   */
  Cone2d(Vector const & p, Vector const & d1, Vector const & d2) { pos= p; dir1= d1; dir2= d2; }

  /** a cone can be specified by his focal point [pos] and two angles
      all vectors belonging to the cone are then

      pos + k * vec; with vec.arg() between [ang1] and [ang2] (going counterclockwise from ang1 to ang2)
   */
  Cone2d(Vector const & p, ANGLE const & ang1, ANGLE const & ang2);

  /** a cone can also be defined by his focal point [pos],its direction,
      and the width of its opening angle

             /
            /
           /
          /
       p /_________  dir
         \
          \
           \
            \
             \


     dir1.norm() is NOT relevant.
   */
   Cone2d(Vector const & p, Vector const & dir, ANGLE const & ang);
   bool inside(Vector const & point) const ;
};

std::ostream& operator<< (std::ostream& o, Set2d const & set);

struct Line2d {
  Line2d(Vector const & p, Vector const & d) { pos= p; dir= d; }
  Vector pos;
  Vector dir;
};

class Geometry2d {
 public:
  static int intersect_circle_with_circle( Vector & p1, Vector & p2, const Circle2d & c1, const Circle2d & c2);

  /*
    Result is false, if the lines are parallel (even if they are identical!!!)
  */
  static bool intersect_lines(Vector & p, Line2d const & L1, Line2d const & L2) {
    double tmp1,tmp2;
    return intersect_lines(p,tmp1,tmp2,L1,L2);
  }
  
  /*
    If the result is true, then
    L1.pos + stretch1 * L1.dir= L2.pos + stretch2 * L2.dir = p
  */
  static bool intersect_lines(Vector & p, double & stretch1, double & stretch2, Line2d const & L1, Line2d const & L2);


  /*
    
  */
  static void projection_to_line(Vector & res, Vector const & pos, Line2d const & line);

  
  static double sqr_distance_to_line(Vector const & pos, Line2d const & line) {
    Vector res;
    projection_to_line(res,pos,line);
    return (pos-res).sqr_norm();
  }

#if 0
   static int intesect_circle_with_x_line(double &x1,double &y1, double &x2, double &y2,
                   double Cx, double Cy, double r,double  xline);

   static int intesect_circle_with_y_line(double &x1,double &y1, double &x2, double &y2,
                   double Cx, double Cy, double r, double yline);

   /* gegeben 2 Punkte (x1,y1) (x2,y2), gesucht sind die beiden Kreise, auf denen die Punkte
      (px,py) liegen, so dass der Winkel zwischen den Vektoren (x1-px,y1-py) und 
      (x2-px,y2-py) den Wert von "angle" hat. Bem: ist der Winkel >90 Grad, so liegen die
      gesuchten Punkte auf dem "laengeren" Bogen des Kreise, wobei der Kreis durch die 
      Strecke von (x1,y1) und (x2,y2) in zwei Bogenabschnitte geteilt wird */
   static int circles_with_angle(double &Cx1, double &Cy1, double &Cx2, double &Cy2,double &radius,
                       double x1, double y1, double x2, double y2, double  angle);
#endif
};


#endif

