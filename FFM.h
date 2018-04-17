#pragma once
#include <Eigen/Core>
#include <memory>
#include<map>
#include<cmath>
using namespace Eigen;
using namespace std;
class FFM {
	typedef shared_ptr<MatrixXd> Matrix;
public:
	FFM();
	FFM(int D);
	~FFM();
	void setAlpaha(double alpha);
	void setBelta(double belta);
	void setL1(double l1);
	void setL2(double l2);
	//简单地映射i->f
	int getF(int i) {

		return i / 16;
	}

	double predict(map<int, double>& m);
	void update(map<int, double>& m, int label, double p);
	void train(map<int, double>& m, int label);
	template<typename ptype>
	double reduce(ptype);
private:
	//meta parameters
	double alpha = 0.025;
	double belta = 1;
	double L1 = 1;
	double L2 = 0.5;
	int D = 10000;
	int k = 4;
	int F = 8;
	//
	//store tmp data F*(K*F)matrix
	Matrix kv;
	//Matrix kvsqr;
	//parameter to train
	double b;
	double bN;
	double bZ;
	//1*n matrix
	Matrix W;
	Matrix WN;
	Matrix WZ;
	//d*(F*k) matrix
	Matrix V;
	Matrix VN;
	Matrix VZ;
	void initial();
	


};
FFM::FFM()
{
	initial();
}
FFM::FFM(int D)
{
	this->D = D;
	initial();
}

FFM::~FFM() {}
void FFM::initial() {
	//new matrix
	kv.reset(new MatrixXd(F, k*F));
	//kvsqr.reset(new MatrixXd(1, k));
	W.reset(new MatrixXd(D, 1));
	WN.reset(new MatrixXd(D, 1));
	WZ.reset(new MatrixXd(D, 1));
	V.reset(new MatrixXd(D, k*F));
	VN.reset(new MatrixXd(D, k*F));
	VZ.reset(new MatrixXd(D, k*F));
	//initialize
	*kv = MatrixXd::Zero(F, k*F);
	//*kvsqr = MatrixXd::Zero(1, k);
	*W = MatrixXd::Zero(D, 1);
	*WN = MatrixXd::Zero(D, 1);
	*WZ = MatrixXd::Zero(D, 1);
	*V = MatrixXd::Zero(D, k*F);
	*VN = MatrixXd::Zero(D, k*F);
	*VZ = MatrixXd::Zero(D, k*F);

}
void FFM::setAlpaha(double alpha) { this->alpha = alpha; };
void FFM::setBelta(double belta) { this->belta = belta; };
void FFM::setL1(double l1) { this->L1 = l1; };
void FFM::setL2(double l2) { this->L2 = l2; };
template<typename ptype>
double FFM::reduce(ptype m) {
	double re = 0.0;
	for (int i = 0; i < m->rows(); i++)
	{
		for (int j = 0; j < m->cols(); j++)
		{
			re += (*m)(i, j);
		}
	}
	return re;
}
double FFM::predict(map<int, double>& m)
{
	double p = b;
	*kv = (*kv) * 0;
	//*kvsqr = (*kvsqr) * 0;
	map<int, double>::iterator iter;
	for (iter = m.begin(); iter != m.end(); iter++)
	{
		int findex = iter->first;
		double value = iter->second;
		int f = getF(findex);
        //# TO DO
		(*kv).row(f) = (*kv).row(f) + (*V).row(findex)*value;
		//*kvsqr = (*kvsqr) + (*V).row(findex).cwiseProduct((*V).row(findex))*value*value;
		p += (*W)(findex, 0);
	}
	for (iter = m.begin(); iter != m.end(); iter++)
	{
		int findex = iter->first;
		double value = iter->second;
		int f = getF(findex);
		//TO DO
		double tmp = 0.0;
		for(int i=0;i<F;i++)
		{
			tmp = tmp + 0.5*value*(((*V).block(findex, i*k, 1, k)*((*kv).block(i, f*k, 1, k).transpose()))(0, 0));
			if (i == f)
			{
				tmp = tmp - 0.5*value*value*((*V).block(findex, i*k, 1, k).cwiseAbs2().sum());
			}
		}
		p += tmp;
	}
	//时间复杂度DFK
	p = 1 / (1 + exp(-p));
	return p;
};
void FFM::update(map<int, double>& m, int label, double p)
{
	double g = p - label;
	double bsigma = (sqrt(bN + g * g) - sqrt(bN)) / alpha;
	bZ += g - bsigma * b;
	bN += g * g;
	int bsign = bZ < 0 ? -1 : 1;
	if (abs(bZ) <= L1) {
		b = 0.0;
	}
	else {
		b = (bsign * L1 - bZ) / ((belta + sqrt(bN)) / alpha + L2);
	}
	map<int, double>::iterator iter;
	for (iter = m.begin(); iter != m.end(); iter++)
	{
		int findex = iter->first;
		double value = iter->second;
		double g2 = g*value;
		double sigma = (sqrt((*WN)(findex, 0) + g2 * g2) - sqrt((*WN)(findex, 0))) / alpha;
		(*WZ)(findex, 0) += g2 - sigma *(*W)(findex, 0);
		(*WN)(findex, 0) += g2 * g2;
		int sign = (*WZ)(findex, 0) < 0 ? -1 : 1;
		if (abs((*WZ)(findex, 0)) <= L1) {
			(*W)(findex, 0) = 0.0;
		}
		else {
			(*W)(findex, 0) = (sign * L1 - (*WZ)(findex, 0)) / ((belta + sqrt((*WN)(findex, 0))) / alpha + L2);
		}
		//
		int f = 0;
		// TO DO
		for (int i = 0; i < F; i++) {
			MatrixXd gk = g2*(*kv).block(i,k*f,1,k);
			if (f == i)
				gk = gk - (*V).block(findex, k*f,1, k);
			MatrixXd  sigmak = (((*VN).block(findex,i*k,1,k) + gk.cwiseProduct(gk)).cwiseSqrt() - (*VN).block(findex, i*k,1,k).cwiseSqrt()) / alpha;
			(*VZ).block(findex, i*k, 1,k) += (gk -sigmak.cwiseProduct((*V).block(findex, i*k, 1, k)));
			(*VZ).block(findex, i*k, 1, k) += gk.cwiseProduct(gk);
			for (int j = 0; j < k; j++)
			{
				int signv = (*VZ)(findex, i*k+j) < 0 ? -1 : 1;
				if (abs((*VZ)(findex, j)) <= L1)
				{
					(*V)(findex, i*k+j) = 0;
				}
				else
				{
					(*V)(findex, i*k+j) = (signv * L1 - (*VZ)(findex, i*k+j)) / ((belta + sqrt((*VN)(findex, i*k+j))) / alpha + L2);
				}
			}
		}

	}

}
void FFM::train(map<int, double>& map, int label)
{
	double p = predict(map);
	update(map, label, p);
};
