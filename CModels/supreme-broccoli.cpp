#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include <cstdio>

extern "C"{
void our_functional_(int* basis_size, double basis[], double weights[],
                    double* sigma, double* rho, double* e_xc, double* v_xc) {

    /*
    for (int i = 0; i < 1; ++i) {
        printf("%f \n", weights[i]);
    }
    printf("\n");
    */
    

    double c_val_d = 1 / std::sqrt(2 * M_PI * std::pow(*sigma, 2));
    double gamma_val_d = -1 / (2 * std::pow(*sigma, 2));

    torch::Tensor t_basis = torch::from_blob(basis, {*basis_size}, torch::kDouble);
    torch::Tensor t_weights = torch::from_blob(weights+1, {*basis_size}, torch::kDouble);
    torch::Tensor t_intercept = torch::from_blob(weights, {1}, torch::kDouble);
    torch::Tensor t_rho = torch::from_blob(rho, {1}, torch::kDouble);

    torch::Tensor out1 = torch::add(t_rho, t_basis, -1);
    torch::Tensor out2 = torch::pow(out1, 2);

    out2.mul_(gamma_val_d);
    out2.exp_();
    out2.mul_(c_val_d);
    out2.mul_(t_weights);

    torch::Tensor t_e_xc = torch::sum(out2) + t_intercept;
    torch::Tensor t_v_xc = torch::dot(out2, out1);
    t_v_xc.mul_(2 * gamma_val_d);

    *e_xc = *((double*) t_e_xc.data_ptr());
    *v_xc = *((double*) t_v_xc.data_ptr());
    

    //printf("%f\n",*weights[0]);

    // LDA 
    
    double A = 0.0311;
    double B = -0.048;
    double C = 0.0020;
    double D = -0.0116;
    double beta1 = 1.0529;
    double beta2 = 0.3334;

    double cx = -0.75 * pow(3.0 / M_PI, 1.0/3.0);
    double pi34 = pow(4.0 * M_PI / 3.0, 1.0/3.0);
    double gamma = -0.1423;
    double rs = 1 / (pow(pi34, 3.0) * pow(*rho, 1.0/3.0));
    double drs = -pow(pi34,3.0)*pow(rs,4.0/3.0)/3.0;

    double ex = cx * pow(*rho, 1.0/3.0);
    double vx = ex + C*pow(*rho, 1.0/3.0)/3.0; 

    double ec = 0.0;
    double vc = 0.0;

    // high density
    if (rs < 1.0) {
        ec = A*log(rs) + B + C*rs*log(rs) + D*rs;
        vc = ec + *rho * (A*drs/rs + C*drs*log(rs) + C*drs + D*drs);
    } else {
    // low density
        ec = gamma / (1.0 + beta1*sqrt(rs) + beta2*rs);
        vc = ec - *rho*pow(ec,2.0)/gamma * (beta1*pow(rs,-1.0/2.0)*drs/2.0 + beta2*drs);
    }

    *e_xc += *rho*(ec + ex) * 1853.25 / (25 * 25 * 25);
    *v_xc += vx + vc * 1853.25 / (25 * 25 * 25);
    

    //printf("Fuck this shit I'm out\n");

}


}
