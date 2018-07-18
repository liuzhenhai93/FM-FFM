import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.util.HashSet;
import java.util.Map;


/**
 * @author by liuzhenhai
 * @date 2018年3月14日
 */
public class FTRLFM {

	private double alpha = 0.025; // learning rate
	private double belta = 1; // smoothing rate
	private double L1 = 1; //
	private double L2 = 0.5; //
	private int D = 1000000;
	private int k = 4;
	// y^=b+sum(wi*xi)+sum(sum<Vi,Vj>xi*xj)

	private double[] KV;
	private double[] KV2;

	private double b;
	private double bN;
	private double bZ;

	private double[][] V;
	private double[][] VN;
	private double[][] VZ;

	private double[] N;
	private double[] Z;
	private double[] W;

	public FTRLFM(int D) {
		this.D = D;
		this.k =3;
		// w
		this.N = new double[D];
		this.Z = new double[D];// cumulated n
		this.W = new double[D];// cumulated w
		// vf
		this.VN = new double[D][];
		this.VZ = new double[D][];
		this.V = new double[D][];
		for (int i = 0; i < D; i++) {
			this.VN[i] = new double[k];
			this.VZ[i] = new double[k];
			this.V[i] = new double[k];
		}

	}

	// a=a+b*c
	public void addm(double[] a, double[] b, double c) {
		for (int i = 0; i < a.length; i++) {
			a[i] = a[i] + b[i] * c;
		}
	}

	public double reduce(double[] a) {
		double re = 0.0;
		for (double d : a)
			re += d;
		return re;
	}

	public double dot(double[] a) {
		double re = 0.0;
		for (double d : a)
			re += d * d;
		return re;
	}

	// a=a+b*b*c*c
	public void addm2(double[] a, double[] b, double c) {
		for (int i = 0; i < a.length; i++) {
			a[i] = a[i] + b[i] * b[i] * c * c;
		}

	}

	public double predict(HashSet<Integer> set) {
		double p = b;
		if (KV2 == null)
			KV2 = new double[k];
		if (KV == null)
			KV = new double[k];

		for (int i = 0; i < k; i++) {
			KV[i] = 0;
			KV2[i] = 0;
		}
		for (int e : set) {
			addm(KV, V[e], 1);
			addm2(KV2, V[e], 1);
			p += W[e];
		}
		p = p + 0.5 * dot(KV) - reduce(KV2);
		p = 1 / (1 + Math.exp(-p));
		return p;
	}

	public void update(HashSet<Integer> set, int label, double p) {
		double g = p - label;

		double bsigma = (Math.sqrt(bN + g * g) - Math.sqrt(bN)) / alpha;
		bZ += g - bsigma * b;
		bN += g * g;
		int bsign = bZ < 0 ? -1 : 1;
		if (Math.abs(bZ) <= L1) {
			b = 0.0;
		} else {
			b = (bsign * L1 - bZ) / ((belta + Math.sqrt(bN)) / alpha + L2);
		}

		for (Integer i : set) {
			// w

			double sigma = (Math.sqrt(N[i] + g * g) - Math.sqrt(N[i])) / alpha;
			Z[i] += g - sigma * W[i];
			N[i] += g * g;
			int sign = Z[i] < 0 ? -1 : 1;
			if (Math.abs(Z[i]) <= L1) {
				W[i] = 0.0;
			} else {
				W[i] = (sign * L1 - Z[i]) / ((belta + Math.sqrt(N[i])) / alpha + L2);
			}
			// v
			for (int j = 0; j < k; j++) {
				double gg = g * (KV[j] - V[i][j]);
				double sigmav = (Math.sqrt(VN[i][j] + g * g) - Math.sqrt(VN[i][j])) / alpha;
				VZ[i][j] += gg - sigmav * V[i][j];
				VN[i][j] += gg * gg;
				int signv = VZ[i][j] < 0 ? -1 : 1;
				if (Math.abs(VZ[i][j]) <= L1) {
					V[i][j] = 0.0;
				} else {
					V[i][j] = (signv * L1 - VZ[i][j]) / ((belta + Math.sqrt(VN[i][j])) / alpha + L2);
				}

			}

		}
	}

	public void train(HashSet<Integer> set, int label) {

		double p = predict(set);
		//System.out.print((p > 0.5) == (label == 1));
		//System.out.println(logloss(p, label));
		update(set, label, p);
		set.clear();
	}

	public double predict(Map<Integer, Double> map) {
		double p = b;
		if (KV2 == null)
			KV2 = new double[k];
		if (KV == null)
			KV = new double[k];

		for (int i = 0; i < k; i++) {
			KV[i] = 0;
			KV2[i] = 0;
		}
		for (int e : map.keySet()) {
			double x = map.get(e);
			addm(KV, V[e], x);
			addm2(KV2, V[e], x);
			p += W[e] * x;
		}
		p = p + 0.5 * (dot(KV) - reduce(KV2));
		p = 1 / (1 + Math.exp(-p));
		return p;
	}

	public void update(Map<Integer, Double> map, int label, double p) {
		double g = p - label;
		double bsigma = (Math.sqrt(bN + g * g) - Math.sqrt(bN)) / alpha;
		bZ += g - bsigma * b;
		bN += g * g;
		int bsign = bZ < 0 ? -1 : 1;
		if (Math.abs(bZ) <= L1) {
			b = 0.0;
		} else {
			b = (bsign * L1 - bZ) / ((belta + Math.sqrt(bN)) / alpha + L2);
		}

		for (Integer i : map.keySet()) {
			// w
			double x = map.get(i);
			double g2 = g * x;
			double sigma = (Math.sqrt(N[i] + g2 * g2) - Math.sqrt(N[i])) / alpha;
			Z[i] += g2 - sigma * W[i];
			N[i] += g2 * g2;
			int sign = Z[i] < 0 ? -1 : 1;
			if (Math.abs(Z[i]) <= L1) {
				W[i] = 0.0;
			} else {
				W[i] = (sign * L1 - Z[i]) / ((belta + Math.sqrt(N[i])) / alpha + L2);
			}
			// v
			for (int j = 0; j < k; j++) {
				double g3 = g2 * (KV[j] - V[i][j] * x);
				double sigmav = (Math.sqrt(VN[i][j] + g3 * g3) - Math.sqrt(VN[i][j])) / alpha;
				VZ[i][j] += g3 - sigmav * V[i][j];
				VN[i][j] += g3 * g3;
				int signv = VZ[i][j] < 0 ? -1 : 1;
				if (Math.abs(VZ[i][j]) <= L1) {
					V[i][j] = 0.0;
				} else {
					V[i][j] = (signv * L1 - VZ[i][j]) / ((belta + Math.sqrt(VN[i][j])) / alpha + L2);
				}

			}

		}
	}

	public void train(Map<Integer, Double> map, int label) {

		double p = predict(map);
		System.out.print((p > 0.5) == (label == 1));
		System.out.println(logloss(p, label));
		update(map, label, p);
		map.clear();
	}

	public double logloss(double p, int label) {
		if(p<0.000000001)
			p=0.000000001;
		if (label == 1) {
			p = -Math.log(p);
		} else {
			p = -Math.log(1.0 - p);
		}
		return p;
	}

	public static void main(String args[]) {
		
		int epoch = 3000; // repeat train times

		FTRLFM ftrl = new FTRLFM(128);

		String trPath = "D:\\sohudev\\xgboost\\demo\\data\\agaricus.txt.train";
		String tePath = "D:\\sohudev\\xgboost\\demo\\data\\agaricus.txt.test";
		String submissionPath = "D:\\tmp\\ftrl\\result\\result.csv";

		BufferedReader br;
		String str = null;
		double p=0.0;
		int label=0;
		double loss=0.0;
		int count=0;
		int rights=0;
		
		try {
			// train model
			for (int epo = 0; epo < epoch; epo++) {
				count=0;
				rights=0;
				loss=0;
				br = new BufferedReader(new InputStreamReader(new FileInputStream(trPath), "UTF-8"));
				//str = br.readLine();
				String[] value = null;
				HashSet<Integer> set = new HashSet<Integer>();

				while ((str = br.readLine()) != null) {
					count+=1;
					value = str.split(" ");
					for (int i = 1; i < value.length; i++) {
						String x = value[i].split(":")[0];
						int xi = Integer.parseInt(x);
						if (xi < 128)
							set.add(xi);
					}
					label=Integer.parseInt(value[0]);
					//ftrl.train(set, Integer.parseInt(value[0]));
					p=ftrl.predict(set);
					loss+=ftrl.logloss(p, label);
					ftrl.update(set, label, p);
					set.clear();
					if ((p > 0.5) == (label == 1))
						rights+=1;
					
				}
				br.close();
				double ac=((double)rights)/count;
				//System.out.println(String.format("train epoch %d:logloss %f,ac %f", epo,loss,ac));
				
				if(epo%50!=1) 
				continue;
				//test model
				count=0;
				rights=0;
				loss=0;
				br = new BufferedReader(new InputStreamReader(new FileInputStream(tePath), "UTF-8"));
				while ((str = br.readLine()) != null) {
					count+=1;
					value = str.split(" ");
					for (int i = 1; i < value.length; i++) {
						String x = value[i].split(":")[0];
						int xi = Integer.parseInt(x);
						if (xi < 128)
							set.add(xi);
					}
					label=Integer.parseInt(value[0]);
					p=ftrl.predict(set);
					//System.out.println("p"+p);
					loss+=ftrl.logloss(p, label)/1000;
					//System.out.println(loss);
					set.clear();
					if ((p > 0.5) == (label == 1))
						rights+=1;
				}
				br.close();
				ac=((double)rights)/count;
				System.out.println(String.format("test epoch %d:logloss %f,ac %f", epo,loss,ac));
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}

/*
 * 
 * FTRL for FM
 *
 * training complexity of(kn),n is the number of feature,k if the dimension of feature latent vector
 *
 *
 */