#ifndef _ANGLE_H_
#define _ANGLE_H_

/*
 * Copyright (c) 1999 - 2000, Artur Merke 
 *
 * This file is part of FrameView2d.
 *
 * FrameView2d is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * FrameView2d is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with FrameView2d; see the file COPYING.  If not, write to
 * the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.
 */

#include <math.h>
#include <iostream>

#include "Vector.h"
class Vector;

class ANGLE {
private:
  /**   Wert des ANGLEs
   *    Der Wert liegtzwischen 0 und (2*PI)
   *    (0.0 <= val < 2*PI)
   *    die Setter-Methode sichert dies ab */
  double val;
  static const double TwoPi;

 public:
  /** Standard-Konstruktor */
  ANGLE();
  /** Kopier-Konstruktor */
  ANGLE(const ANGLE &otherAng);
  /** Konstruktor (setzt den internen Wert auf den Wert von newVal) */
  ANGLE(double newVal);
  /** Konstruktor (berechnet den ANGLE aus Vector Koordinaten) */
  ANGLE(double xVal, double yVal);
  /** Konstruktor (berechnet den ANGLE aus Vector) */
  ANGLE(const Vector &vec);

  /** Destruktor */
  virtual ~ANGLE();


  /** Setter-Method */
  void set_value(double newVal);
  /** Getter-Method - gibt val zurück (0.0 <= val < 2*PI) */
  double get_value() const;
  /** Getter-Method - gibt val zurück (0.0 <= val < 2*PI) (Selbe wie get_value())*/
  double get_value_0_p2PI() const;
  /** Getter-Method - gibt val zurück (-PI < val <= PI) */
  double get_value_mPI_pPI() const;


  /** Berechnet die Differenz zwischen 2 ANGLEs */
  double diff(const ANGLE &otherAng) const;


  ANGLE& operator=(const ANGLE &otherAng);
  void operator+=(const ANGLE &otherAng);
  void operator-=(const ANGLE &otherAng);
  void operator*=(const double &multiplier);
  void operator/=(const double &divisor);

  ANGLE operator+() const;
  ANGLE operator-() const;
  ANGLE operator+(const ANGLE &otherAng) const;
  ANGLE operator-(const ANGLE &otherAng) const;
  ANGLE operator*(const double &multiplier) const;
  friend ANGLE operator*(const double &multiplier, const ANGLE &ang);
  ANGLE operator/(const double &divisor) const;
  friend ANGLE operator/(const double &divisor, const ANGLE &ang);

  bool operator==(const ANGLE &otherAng) const;
  bool operator!=(const ANGLE &otherAng) const;
  bool operator<( const ANGLE &otherAng) const;
  bool operator>( const ANGLE &otherAng) const;
  bool operator<=(const ANGLE &otherAng) const;
  bool operator>=(const ANGLE &otherAng) const;

  friend std::istream& operator>>(std::istream& inStream, ANGLE &ang);
};

std::ostream& operator<<(std::ostream& outStream, const ANGLE &ang);
double sin(const ANGLE &ang);
double cos(const ANGLE &ang);

#endif

