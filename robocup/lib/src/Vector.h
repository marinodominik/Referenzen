#ifndef _VECTOR_H_
#define _VECTOR_H_

#include <math.h>
#include <iostream>

#include "angle.h"
class ANGLE;

class Vector
{
private:
    double x;
    double y;

public:
    double  getX() const;
    void    setX(     double newX );

    Vector& addToX(   double summand );
    Vector& subFromX( double subtrahend );
    Vector& mulXby(   double multiplier );
    Vector& divXby(   double divisor );

    double  getY() const;
    void    setY(     double newY );

    Vector& addToY(   double summand );
    Vector& subFromY( double subtrahend );
    Vector& mulYby(   double multiplier );
    Vector& divYby(   double divisor );

    void    setXY(    double newX, double newY );

    void clone( const Vector &origin );



    Vector();
    Vector( const Vector &origin );
    Vector( const double vx, const double vy );
    Vector( const ANGLE & ang );

    Vector& operator=( Vector vec );

    Vector operator+( const Vector &otherVec ) const;
    Vector operator-( const Vector &otherVec ) const;
    Vector operator*( const double &multiplier ) const;
    friend Vector operator*( const double &multiplier, const Vector &vec );
    Vector operator/( const double &divisor ) const;

    Vector operator-() const;

    void operator+=( Vector vec );
    void operator-=( Vector vec );
    void operator*=( double val );
    void operator/=( double val );

    /* Vergleichsoperatoren zum Vergleichen der Länge zweier Vektoren */
    bool operator<=( const Vector &otherVec ) const;
    bool operator<(  const Vector &otherVec ) const;
    bool operator>=( const Vector &otherVec ) const;
    bool operator>(  const Vector &otherVec ) const;

    /* Testet ob der aufrufende Vector zwischen minVec und maxVec liegt */
    bool isBetween( Vector minVec, Vector maxVec ) const;

    /** Gibt den euklidischen Abstand zwischen den Vektoren zurück */
    double distance(     const Vector& orig ) const;

    /** Gibt den quadratische euklidischen Abstand zwischen den Vektoren zurück.
     * Diese Routine is effizienter als distance(...), da das Wurzelziehen entfaellt!!! */
    double sqr_distance( const Vector& orig ) const;

    /** gibt den Winkel zwischen dem Vector selbst und dem Vector dir */
    double angle(   const Vector& dir ) const;
    ANGLE  ANGLE_to( const Vector& dir ) const;

    /** rotiert um den angegebenen Winkel */
    Vector& rotate( const double& ang );
    Vector& ROTATE( const ANGLE& ang );

    double norm() const;
    double sqr_norm() const;
    Vector& normalize( double l = 1.0 );

    /** berechnet das Argument der komplexen Zahl (x,y) auf das Intervall
     [0,2*PI] normalisiert. Das Argument von (0,0) wird auf 0 gesetzt*/
    double arg() const;
    ANGLE  ARG() const;

    double angle();

    /** berechnet eine reelle Zahl modulo 2*PI, so dass der normalisierte
     Winkel im Intervall [0,2*PI) liegt */
    double normalize_angle( const double& a ) const;

    /** berechnet das Skalarprodukt Zwischen 2 Vektoren */
    double dot_product( const Vector& orig ) const;

    Vector& init_polar( const double& n, const double& a );
    Vector& init_polar( const double& n, const ANGLE& a );
};

extern std::ostream& operator<<( std::ostream &outStream, const Vector &vector );

#endif
