
#include "SpecialFunc.h"

// Intel math function
#include <mathimf.h>



// wrap Intel ERF
double myerf(double x) {
	return erf(x);
}

// wrap Intel ERFC
double myerfc(double x) {
	return erfc(x);
}



