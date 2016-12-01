import javax.swing.*;
import javax.swing.border.BevelBorder;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.util.ArrayList;

/**
 * Created by Владимир on 01.12.2016.
 */
public class View {

    private JFrame mainWindow = new JFrame("ML-2");
    private int frameWidth = 405;
    private int frameHeight = 450;
    private int firstLocation = 30;
    private JLabel statusBar = new JLabel("AUC:");
    private JLabel canvas = new JLabel();
    private BufferedImage image = new BufferedImage(400, 400, BufferedImage.TYPE_4BYTE_ABGR);

    View(ArrayList<Point> ps, double auc)
    {
        mainWindow.setSize(frameWidth, frameHeight);
        mainWindow.setLocation(firstLocation, firstLocation);
        mainWindow.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        mainWindow.setLayout(new BorderLayout());

        statusBar.setBorder(new BevelBorder(BevelBorder.LOWERED));
        mainWindow.add(statusBar, BorderLayout.SOUTH);
        mainWindow.add(canvas, BorderLayout.NORTH);
        mainWindow.setResizable(false);

        canvas.setIcon(new ImageIcon(image));
        Graphics gr = image.getGraphics();
        gr.setColor(Color.WHITE);
        gr.fillRect(0, 0, 400, 400);

        // draw x and y axis
        gr.setColor(Color.BLACK);
        gr.drawLine(20, 370, 20, 10);
        gr.drawLine(20, 370, 380, 370);

        gr.drawLine(20, 10, 17, 15);
        gr.drawLine(20, 10, 23, 15);

        gr.drawLine(380, 370, 375, 367);
        gr.drawLine(380, 370, 375, 373);

        gr.drawString("FPR", 370, 385);
        gr.drawString("TPR", 0, 10);

        // draw ROC-curve
        gr.setColor(Color.BLUE);
        int px = 370, py = 20;
        for (Point p: ps) {
            int x = (int) (350 * p.FPR) + 20;
            int y = (int) (350 * (1 - p.TPR)) + 20;
            //gr.fillOval(x, y, 3, 3);
            gr.drawLine(px, py, x, y);
            px = x;
            py = y;
        }

        statusBar.setText("AUC: " + auc);

        mainWindow.setVisible(true);
        drawPoints(ps);
    }

    private void drawPoints(ArrayList<Point> ps) {

    }
}
