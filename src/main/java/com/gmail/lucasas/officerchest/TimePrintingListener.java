package com.gmail.lucasas.officerchest;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;

public class TimePrintingListener extends BaseTrainingListener {

    private long lastReportTime = 0;
    private long reportIntervalMs; // e.g., 2 minutes = 120000 ms

    public TimePrintingListener(long reportIntervalMs) {
        this.reportIntervalMs = reportIntervalMs;
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        long now = System.currentTimeMillis();
        if (now - lastReportTime >= reportIntervalMs) {
            lastReportTime = now;
            System.out.println("TimePrintingListener: epoch=" + epoch
                    + ", iteration=" + iteration
                    + ", score=" + model.score());
        }
    }
}
