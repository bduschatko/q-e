#include <cmath>
#include <string>
#include <iostream>

extern "C"
{
void torch_model_(double **params, double *rho, double *exc_custom, double *vxc_custom)
{
    double pi34 = 0.6203504908994;
    double f = -0.687247939924714;
    double alpha = 2.0/3.0;
    double pisq = 9.86960440109;
    double rs = pi34 / pow(*rho, 1.0/3.0);

    double vx = 0.0;
    double vc = 0.0;
    double ex = 0.0;
    double ec = 0.0;

    ex = f * alpha / rs; 
    vx = 4.0 / 3.0 * f * alpha / rs; 

    *exc_custom = ex + ec;
    *vxc_custom = vx + vc;

    // std::cout << *params << std::endl;
    double* local_params = *params;
    printf("%f\n", local_params[0]);
    printf("%f\n", local_params[1]);
    
}
}
