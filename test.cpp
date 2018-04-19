#pragma warning(disable:4996)
#include <iostream>
#include <fstream>
#include<vector>
#include <string>
#include <string.h>
#include <map>
#include"FM.h"
#include "FFM.h"
using namespace std;


vector<string> split(const string& str, const string& delim) {
	vector<string> res;
	if ("" == str) return res;
 
	char * strs = new char[str.length() + 1];
	strcpy(strs, str.c_str());

	char * d = new char[delim.length() + 1];
	strcpy(d, delim.c_str());

	char *p = strtok(strs, d);
	while (p) {
		string s = p; 
		res.push_back(s); 
		p = strtok(NULL, d);
	}
	return res;
}
 double logloss(double p, int label) {
	if (p<0.000000001)
		p = 0.000000001;
	if (label == 1) {
		p = -log(p);
	}
	else {
		p = -log(1.0 - p);
	}
	return p;
}
 template <typename Model>
 void oneEpoch(bool istrain, string file, Model & mymodel, int & rights, int & count, double& loss);
 template<typename Model>
 void oneEpoch(bool istrain,string file,Model & mymodel,int & rights,int & count,double& loss)
 {
	 //train
	 count = 0;
	 rights = 0;
	 loss = 0;
	 string tmp;
	 ifstream trfile(file);
	 map<int, double> m;
	 while (getline(trfile, tmp))
	 {
		 count++;
		 vector<string> fs = split(tmp, " ");
		 int label = stoi(fs[0]);
		 for (int i = 1; i < fs.size(); i++)
		 {
			 vector<string> indexs = split(fs[i], ":");
			 int findex = stoi(indexs[0]);
			 double value = stod(indexs[1]);
			 if (findex < 128)
				 m.insert(pair<int, double>(findex, value));
		 }
		double p = mymodel.predict(m);
		 loss += logloss(p, label);
		 if(istrain)
		 mymodel.update(m, label, p);
		 m.clear();
		 if ((p > 0.5) == (label == 1))
			 rights += 1;

	 }
	 trfile.close();
 }
 template<typename Model>
 void testfm();
 template<typename Model>
void testfm()
{
	int epoch = 30;
	string trpath = "D:\\xgboost\\demo\\data\\agaricus.txt.train";
	string tepath = "D:\\xgboost\\demo\\data\\agaricus.txt.test";
	//ifstream myfile("G:\\C++ project\\Read\\hello.txt");
	//if (!myfile.is_open())
	//{
	//	cout << "未成功打开文件" << endl;
	//}
	Model mymodel(128);
	double loss = 0.0;
	int count = 0;
	int rights = 0;
	try {
		for (int epo = 0; epo<epoch; epo++)
		{
			oneEpoch(true, trpath, mymodel, rights, count, loss);
			if (epo % 5 == 3) {
				cout << "train-" << epo<<":"<<((double)rights) / count << endl;
			}
			//test
			
			oneEpoch(false, tepath, mymodel, rights, count, loss);
			
			if (epo % 10 == 2) {
				cout << "test-" << epo << ":" << ((double)rights) / count << endl;
			}
		}
		int noop;
		cin >> noop;
	}
	catch (exception& e)
	{
		cout << e.what();
	}
}
int main(int, char *[])
{
	testfm<FFM>();
}
