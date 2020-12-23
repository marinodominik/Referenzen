// 2-dimensionale Normalverteilung mit pdf- und Zufallsfunktion
// Spezialanfertigung fuer die Brainstormers
// created 25-APR-2003 by Martin Lauer
// ------------------------------------------------------------

#ifndef normal_dist2d_h
#define normal_dist2d_h

//#include <stdexcept>

class Normal_distribution_2d {
 public:
  Normal_distribution_2d (double, double, double);// throw (invalid_argument);
  // Argumente: (arg1, arg2)=Varianzen, arg3=Kovarianz

  void operator() (double&, double&, double, double) const;// throw ();
  // Zufallszahlenerzeuger, liefert die Zufallswerte in (arg1, arg2)

  double pdf (double, double, double, double) const;// throw ();
  // Dichtefunktion

 protected:
  double c1, c2, c3;  // Cholesky-Faktor
  double ic1, ic2, ic3;  // Inverser Cholesky-Faktor
  double normal;  // Normalisierungskonstante
};

#endif
