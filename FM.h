#pragma once
#include <Eigen/Core>
#include <memory>
#include<map>
#include<cmath>
using namespace Eigen;
using namespace std;

class FM
{
public:
	typedef shared_ptr<MatrixXd> Matrix;
	FM();
	FM(int D);
	FM(int D, int k);
	~FM();
	void setAlpaha(double alpha);
	void setBelta(double belta);
	void setL1(double l1);
	void setL2(double l2);
	
	double predict(map<int, double>& m);
	void update(map<int,double>& m,int label,double p);
	void train(map<int, double>& m,int label);
	template<typename ptype>
	double reduce(ptype);
	
private:
	//meta parameters
	double alpha = 0.025;
	double belta = 1;
	double L1 = 1;
	double L2 = 0.5;
	int D = 10000;
	int k = 8;
	//store tmp data 1*K matrix
	Matrix kv;
	Matrix kvsqr;
	//parameter to train
	double b;
	double bN;
	double bZ;
	//1*n matrix
	Matrix W;
	Matrix WN;
	Matrix WZ;
	//d*k matrix
	Matrix V;
	Matrix VN;
	Matrix VZ;
	void initial();

};

FM::FM()
{
	initial();

}
FM::FM(int D)
{
	this->D = D;
	initial();
}
FM::FM(int D,int k)
{
	this->D = D;
	this->k = k;
	initial();
}
//initial after metaparameter is set
void FM::initial()
{  
	//new matrix
	kv.reset(new MatrixXd(1, k));
	kvsqr.reset(new MatrixXd(1, k));
	W.reset(new MatrixXd(D,1));
	WN.reset(new MatrixXd(D,1));
	WZ.reset(new MatrixXd(D,1));
	V.reset(new MatrixXd(D, k));
	VN.reset(new MatrixXd(D, k));
	VZ.reset(new MatrixXd(D, k));
	//initialize
	*kv = MatrixXd::Zero(1, k);
	*kvsqr = MatrixXd::Zero(1, k);
	*W = MatrixXd::Zero(D, 1);
	*WN = MatrixXd::Zero(D, 1);
	*WZ = MatrixXd::Zero(D, 1);
	*V = MatrixXd::Zero(D, k);
	*VN = MatrixXd::Zero(D, k);
	*VZ = MatrixXd::Zero(D, k);


}

void FM::setAlpaha(double alpha) { this->alpha = alpha; };
void FM::setBelta(double belta) { this->belta = belta; };
void FM::setL1(double l1) { this->L1 = l1; };
void FM::setL2(double l2) { this->L2 = l2; };
template<typename ptype>
double FM::reduce(ptype m) {
	double re = 0.0;
	for (int i =0; i < m->rows(); i++)
	{
		for (int j = 0; j < m->cols(); j++)
		{
			re += (*m)(i,j);
		}
	}
	return re;
}
double FM::predict(map<int, double>& m)
{
	double p = b;
	*kv = (*kv) * 0;
	*kvsqr = (*kvsqr) * 0;
	map<int, double>::iterator iter;
	for(iter=m.begin() ;iter!=m.end();iter++)
	{
		int findex = iter->first;
		double value= iter->second;
		*kv = (*kv) + (*V).row(findex)*value;
		*kvsqr = (*kvsqr) + (*V).row(findex).cwiseProduct((*V).row(findex))*value*value;
		p += (*W)(findex, 0);
	}
	MatrixXd t= (*kv).cwiseProduct(*kv);
	p = p + 0.5*(reduce(&t) - reduce(kvsqr));
	p = 1 / (1 + exp(-p));
	return p;
};
void FM::update(map<int, double>& m,int label,double p)
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
		double sigma = (sqrt((*WN)(findex,0) + g2 * g2) - sqrt((*WN)(findex,0))) / alpha;
		(*WZ)(findex,0) += g2 - sigma *(* W)(findex,0);
		(*WN)(findex,0) += g2 * g2;
		int sign = (*WZ)(findex,0) < 0 ? -1 : 1;
		if (abs((*WZ)(findex,0)) <= L1) {
			(*W)(findex,0) = 0.0;
		}
		else {
			(*W)(findex, 0) = (sign * L1 - (*WZ)(findex,0)) / ((belta + sqrt((*WN)(findex,0))) / alpha + L2);
		}
		MatrixXd gk = g2*((*kv) - (*V).row(findex)*value);
		MatrixXd  sigmak = (((*VN).row(findex) + gk.cwiseProduct(gk)).cwiseSqrt()-(*VN).row(findex).cwiseSqrt()) / alpha;
		(*VZ).row(findex) += gk - sigmak.cwiseProduct((*V).row(findex));
		(*VZ).row(findex) += gk.cwiseProduct(gk);
		for (int j = 0; j < k; j++)
		{
			int signv = (*VZ)(findex, j) < 0 ? -1 : 1;
			if(abs((*VZ)(findex,j))<=L1)
			{
				(*V)(findex, j) = 0;
			}else
			{
				(*V)(findex,j)= (signv * L1 - (*VZ)(findex,j)) / ((belta + sqrt((*VN)(findex,j))) / alpha + L2);
			}
		}

	}

};
void FM::train(map<int, double>& map,int label) 
{
	double p = predict(map);
	update(map, label, p);
};

FM::~FM()
{

}

