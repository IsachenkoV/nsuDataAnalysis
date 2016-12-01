#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <time.h>

using namespace std;

#define mp make_pair

const int CLASSES = 3;
const int PCOUNT = 4;
const int SZ = 150;
const int TESTSZ = 30;
const int TRAINSZ = 120;
const int EPS = 12;

struct Iris
{
	double Data[PCOUNT];
	int Ans;
};

Iris RawData[SZ], TestData[TESTSZ], TrainData[TRAINSZ];
int ConfusionMatrix[CLASSES][CLASSES];
int MethodAns[TESTSZ];
double Recall[CLASSES], Precision[CLASSES];

// Euclid distance
double dist(const Iris& a, const Iris& b)
{
	double result = 0.0;
	for (int i = 0; i < PCOUNT; i++)
	{
		double temp = a.Data[i] - b.Data[i];
		result += temp*temp;
	}
	return sqrt(result);
}

// Evaluations based on MethodAns[] array
#pragma region Evaluations

void getRecall()
{
	for (int i = 0; i < CLASSES; i++)
	{
		double tp = ConfusionMatrix[i][i];
		double tpfn = 0;
		for (int j = 0; j < CLASSES; j++)
			tpfn += ConfusionMatrix[i][j];

		Recall[i] = tp / tpfn;
	}
}

void getPrecision()
{
	for (int i = 0; i < CLASSES; i++)
	{
		double tp = ConfusionMatrix[i][i];
		double tpfp = 0;
		for (int j = 0; j < CLASSES; j++)
			tpfp += ConfusionMatrix[j][i];

		Precision[i] = tp / tpfp;
	}
}

void getConfusionMatrix()
{
	for (int i = 0; i < CLASSES; i++)
		for (int j = 0; j < CLASSES; j++)
			ConfusionMatrix[i][j] = 0;

	for (int i = 0; i < TESTSZ; i++)
	{
		int x = TestData[i].Ans;
		int y = MethodAns[i];
		ConfusionMatrix[x][y]++;
	}
}

void getEvaluation()
{
	getConfusionMatrix();
	getRecall();
	getPrecision();
}

void printEvaluation()
{
	cout << "Confusion Matrix:\n";
	for (int i = 0; i < CLASSES; i++)
	{
		for (int j = 0; j < CLASSES; j++)
		{
			cout << ConfusionMatrix[i][j] << " ";
		}
		cout << "\n";
	}

	cout << "Recall:\n";
	for (int i = 0; i < CLASSES; i++)
		printf("%.5lf ", Recall[i]);
	cout << "\n";

	cout << "Precision:\n";
	for (int i = 0; i < CLASSES; i++)
		printf("%.5lf ", Precision[i]);
	cout << "\n\n";
}

#pragma endregion

#pragma region Methods

void knn(int k)
{
	cout << "k-nearest-neighbors\n";
	int vote[CLASSES];

	for (int i = 0; i < TESTSZ; i++)
	{
		memset(vote, 0, sizeof(vote));

		// get first k nearest point
		vector < pair < double, int > > best;
		for (int j = 0; j < TRAINSZ; j++)
		{
			best.push_back(mp(dist(TestData[i], TrainData[j]), j));
		}
		sort(best.begin(), best.end());

		for (int j = 0; j < min(k, TRAINSZ); j++)
		{
			int ind = best[j].second;
			vote[TrainData[ind].Ans]++;
		}

		// find argmax
		int cur_max = -1;
		for (int j = 0; j < CLASSES; j++)
		{
			if (vote[j] > cur_max)
			{
				cur_max = vote[j];
				MethodAns[i] = j;
			}
		}
	}
}

void wknn(int k)
{
	cout << "weighted k-nearest-neighbors\n";
	double wvote[CLASSES];

	for (int i = 0; i < TESTSZ; i++)
	{
		memset(wvote, 0.0, sizeof(wvote));
		// get first k nearest point
		vector < pair < double, int > > best;
		for (int j = 0; j < TRAINSZ; j++)
		{
			double d = dist(TestData[i], TrainData[j]);
			best.push_back(mp(d, j));
		}
		sort(best.begin(), best.end());

		for (int j = 0; j < min(k, TRAINSZ); j++)
		{
			int ind = best[j].second;
			wvote[TrainData[ind].Ans] += 1 / (best[j].first * best[j].first);
		}

		// find argmax
		double cur_max = -1.0;
		for (int j = 0; j < CLASSES; j++)
		{
			if (wvote[j] > cur_max)
			{
				cur_max = wvote[j];
				MethodAns[i] = j;
			}
		}
	}
}

double parzenKernel(double r)
{
	if (r < 0.0 || r > 1.0)
		return 0.0;
	return 0.75 * (1.0 - r*r);
}

void parzen()
{
	cout << "Parzen window method\n";
	// get optimal h with one-leave-out cross-validation
	double curH = 0.01, maxH = 3, delta = 0.001;
	int maxOkAns = 0;
	double optimalH = 0.0;
	while (curH < maxH)
	{
		int sum = 0;
		for (int oloInd = 0; oloInd < TRAINSZ; oloInd++)
		{
			double wvote[CLASSES];
			fill(wvote, wvote + CLASSES, 0.0);
			for (int i = 0; i < TRAINSZ; i++)
			{
				if (i == oloInd) continue;
				wvote[TrainData[i].Ans] += parzenKernel(dist(TrainData[i], TrainData[oloInd]) / curH);
			}

			double cur_max = -1.0;
			int ansClass = -1;
			for (int i = 0; i < CLASSES; i++)
			{
				if (wvote[i] > cur_max)
				{
					cur_max = wvote[i];
					ansClass = i;
				}
			}

			if (ansClass == TrainData[oloInd].Ans)
				sum++;
		}

		if (sum > maxOkAns)
		{
			maxOkAns = sum;
			optimalH = curH;
		}
		curH += delta;
	}

	// with finded optimal h get method result
	for (int i = 0; i < TESTSZ; i++)
	{
		double wvote[CLASSES];
		fill(wvote, wvote + CLASSES, 0.0);
		for (int j = 0; j < TRAINSZ; j++)
		{
			wvote[TrainData[j].Ans] += parzenKernel(dist(TrainData[j], TestData[i]) / optimalH);
		}

		// argmax
		double cur_max = -1.0;
		for (int j = 0; j < CLASSES; j++)
		{
			if (wvote[j] > cur_max)
			{
				cur_max = wvote[j];
				MethodAns[i] = j;
			}
		}
	}
}

double potentialKernel(double r)
{
	return (1.0 / (1.0 + r));
}

void potential()
{
	cout << "Potential functions method\n";
	// fixed parameter H
	const double H = 2.2;

	// initially gammas = 0
	int gamma[TRAINSZ];
	fill(gamma, gamma + TRAINSZ, 0);
	
	// setting parameters
	int errCnt;
	int curInd = -1;
	while (true)
	{
		// count errors
		errCnt = 0;
		for (int i = 0; i < TRAINSZ; i++)
		{
			double wvote[CLASSES];
			fill(wvote, wvote + CLASSES, 0.0);
			for (int j = 0; j < TRAINSZ; j++)
			{
				if (i == j) continue;
				wvote[TrainData[j].Ans] += gamma[j] * parzenKernel(dist(TrainData[j], TrainData[i]) / H);
			}

			double cur_max = -1.0;
			int cur_ans = -1;
			for (int j = 0; j < CLASSES; j++)
			{
				if (wvote[j] > cur_max)
				{
					cur_max = wvote[j];
					cur_ans = j;
				}
			}

			if (cur_ans != TrainData[i].Ans)
			{
				errCnt++;
			}
		}

		if (errCnt < EPS) // EPS ~ 10% * TRAINSZ
			break;

		curInd = (curInd + 1) % TRAINSZ;
		{
			double wvote[CLASSES];
			fill(wvote, wvote + CLASSES, 0.0);
			for (int j = 0; j < TRAINSZ; j++)
			{
				if (curInd == j) continue;
				wvote[TrainData[j].Ans] += gamma[j] * parzenKernel(dist(TrainData[j], TrainData[curInd]) / H);
			}

			double cur_max = -1.0;
			int cur_ans = -1;
			for (int j = 0; j < CLASSES; j++)
			{
				if (wvote[j] > cur_max)
				{
					cur_max = wvote[j];
					cur_ans = j;
				}
			}

			// update gamma[ind]
			if (cur_ans != TrainData[curInd].Ans)
			{
				gamma[curInd]++;
			}
		}
	}

	// with finded gammas and fexed H get method result	
	for (int i = 0; i < TESTSZ; i++)
	{
		double wvote[CLASSES];
		fill(wvote, wvote + CLASSES, 0.0);
		for (int j = 0; j < TRAINSZ; j++)
		{
			wvote[TrainData[j].Ans] += gamma[j] * parzenKernel(dist(TrainData[j], TestData[i]) / H);
		}

		// argmax
		double cur_max = -1.0;
		for (int j = 0; j < CLASSES; j++)
		{
			if (wvote[j] > cur_max)
			{
				cur_max = wvote[j];
				MethodAns[i] = j;
			}
		}
	}
}

#pragma endregion

int main()
{
	// reading data
	freopen("RawIrisData.txt", "r", stdin);
	freopen("output.txt", "w", stdout);
	for (int i = 0; i < SZ; i++)
	{
		for (int j = 0; j < PCOUNT; j++)
			cin >> RawData[i].Data[j];
		cin >> RawData[i].Ans;
	}

	// shuffle and construct train/test sets
	srand(time(NULL));
	random_shuffle(RawData, RawData + SZ);

	int k[3] = { 0, 0, 0 };
	int curtr = 0, curts = 0;
	for (int i = 0; i < SZ; i++)
	{
		int to = RawData[i].Ans;
		if (k[to] < TESTSZ / 3)
		{
			k[to]++;
			TestData[curts] = RawData[i];
			curts++;
		}
		else
		{
			TrainData[curtr] = RawData[i];
			curtr++;
		}
	}

	//run methods

	knn(1);
	getEvaluation();
	printEvaluation();

	wknn(1);
	getEvaluation();
	printEvaluation();

	parzen();
	getEvaluation();
	printEvaluation();

	potential();
	getEvaluation();
	printEvaluation();

	return 0;
}