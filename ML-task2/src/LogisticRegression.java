import com.sun.org.apache.bcel.internal.generic.ARRAYLENGTH;

import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;

/**
 * Created by Владимир on 30.11.2016.
 */
public class LogisticRegression {

    private static final String FILE_NAME = "resources//RawIrisData.txt";
    private static final int DATA_SIZE = 150;
    private static final int TRAIN_SIZE = 60;
    private static final int TEST_SIZE = 90;
    private static final int N = 60;
    private static final int P = 30;
    private static final int MAX_ITERATIONS = 1000;
    private static final double ALPHA = 0.001;

    static ArrayList<Iris> allData = new ArrayList<>();
    static ArrayList<Iris> trainData = new ArrayList<>();
    static ArrayList<Iris> testData = new ArrayList<>();
    static double[] theta;

    static ArrayList<Point> ps = new ArrayList<>();

    private static void readData()
    {
        try (Scanner br = new Scanner(new File(FILE_NAME)))
        {
            for (int i = 0; i < DATA_SIZE; i++)
            {
                Iris t = new Iris();
                for (int j = 0; j < 4; j++)
                    t.x[j] = br.nextDouble();
                t.ans = br.nextDouble();
                allData.add(t);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 50; j++)
            {
                Iris t = allData.get(i * 50 + j);
                if (i == 1)
                    t.ans = 1;
                else
                    t.ans = 0;
                if (j < 20)
                    trainData.add(t);
                else
                    testData.add(t);
            }
        }
    }

    private static void init()
    {
        theta = new double[4];
        for (int i = 0; i < 4; i++)
            theta[i] = 0.0;
    }

    private static double sigmoid(double z)
    {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    private static double scalarMul(double[] a, double[] b)
    {
        double res = 0.0;
        for (int i = 0; i < 4; i++)
            res += a[i]*b[i];
        return res;
    }

    private static void trainRegression() {
        for (int it = 0; it < MAX_ITERATIONS; it++)
        {
            double[] cur_theta = new double[4];
            System.arraycopy(theta, 0, cur_theta, 0, 4);

            for (int j = 0; j < 4; j++)
            {
                double sum = 0.0;
                for (int i = 0; i < TRAIN_SIZE; i++)
                {
                    Iris e = trainData.get(i);
                    double cur_ans = sigmoid(scalarMul(cur_theta, e.x));
                    sum += (e.ans - cur_ans) * e.x[j];
                }
                sum = sum / (TRAIN_SIZE + 0.0);
                sum *= ALPHA;

                theta[j] = cur_theta[j] + sum;
            }
        }
    }

    static double[] classify(double rate)
    {
        double[] results = new double[TEST_SIZE];

        for (int i = 0; i < TEST_SIZE; i++)
        {
            Iris obj = testData.get(i);

            double ans = sigmoid(scalarMul(theta, obj.x));
            if (ans >= rate)
                results[i] = 1.0;
            else
                results[i] = 0.0;
        }

        return results;
    }

    static void getAccuracy()
    {
        for (double rate = 0.0; rate <= 1.0; rate += 0.0005)
        {
            double[] cur_res = classify(rate);
            Point p = new Point();
            double fp = 0, tp = 0;
            for (int i = 0; i < TEST_SIZE; i++)
            {
                double right_ans = testData.get(i).ans;
                if (cur_res[i] == 1)
                    if (right_ans == 1)
                        tp += 1.0;
                    else
                        fp += 1.0;
            }

            p.rate = rate;
            p.FPR = fp / (N + 0.0);
            p.TPR = tp / (P + 0.0);
            ps.add(p);
        }
    }

    private static double getAuc() {
        double result = 0.0;

        for (int i = 1; i < ps.size(); i++)
        {
            Point prev = ps.get(i - 1);
            Point cur = ps.get(i);

            result += Math.abs(prev.FPR - cur.FPR) * (prev.TPR + cur.TPR) / 2.0;
        }
        return result;
    }

    public static void main(String args[])
    {
        readData();
        init();
        trainRegression();

        getAccuracy();

        View v = new View(ps, getAuc());
    }
}
