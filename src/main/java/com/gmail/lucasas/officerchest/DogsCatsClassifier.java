package com.gmail.lucasas.officerchest;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.awt.*;
import java.awt.datatransfer.DataFlavor;
import java.awt.dnd.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DogsCatsClassifier extends JFrame {

    private static final long serialVersionUID = 1L;

    private static final int HEIGHT = 128;
    private static final int WIDTH = 128;
    private static final int CHANNELS = 3;

    private static final int BATCH_SIZE = 128;
    private static final long SEED = 1234;
    private static final double LEARNING_RATE = 0.0003;
    private static final double L2_REG = 0.00001;
    private static final int MAX_EPOCHS = 8;

    private static final String TRAIN_FOLDER = "C:/Users/lucas/data/train";
    private static final String VAL_FOLDER = "C:/Users/lucas/data/val";

    private static final String SAVED_MODEL_FILENAME = "savedModel.zip";

    private static MultiLayerNetwork model;

    private JLabel imageLabel;
    private JLabel predictionLabel;
    private JButton buttonLoadImage;
    private JFileChooser fileChooser;

    public static void main(String[] args) throws Exception {
        System.setProperty("org.bytedeco.javacpp.maxbytes", "16G");
        System.setProperty("org.bytedeco.javacpp.maxphysicalbytes", "16G");

        File savedModel = new File(SAVED_MODEL_FILENAME);
        if (savedModel.exists()) {
            System.out.println("En gemt model fundet! Loader model fra " + SAVED_MODEL_FILENAME + " ...");
            model = MultiLayerNetwork.load(savedModel, false);
        } else {
            System.out.println("Ingen gemt model fundet. Bygger og træner en ny model...");

            model = buildModel();


            DataSetIterator[] iterators = prepareDataIterators();
            DataSetIterator trainIter = iterators[0];
            DataSetIterator valIter   = iterators[1];


            trainModelAndTrackLoss(model, trainIter, valIter);


            System.out.println("Træning færdig! Gemmer model til " + SAVED_MODEL_FILENAME);
            model.save(savedModel);
        }


        evaluateOnValidationSet(model);


        SwingUtilities.invokeLater(() -> {
            DogsCatsClassifier gui = new DogsCatsClassifier();
            gui.setVisible(true);
        });
    }


    private static MultiLayerNetwork buildModel() {
        MultiLayerConfiguration conf = new org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder()
                .seed(SEED)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .l2(L2_REG)
                .updater(new Adam(LEARNING_RATE))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new ConvolutionLayer.Builder(3, 3)
                        .nIn(CHANNELS)
                        .stride(1, 1)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(3, new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(4, new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(6, new BatchNormalization.Builder().build())
                .layer(7, new DenseLayer.Builder().nOut(256).activation(Activation.RELU).build())
                .layer(8, new DropoutLayer.Builder(0.5).build())
                .layer(9, new DenseLayer.Builder().nOut(128).activation(Activation.RELU).build())
                .layer(10, new DropoutLayer.Builder(0.5).build())
                .layer(11, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(2)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(HEIGHT, WIDTH, CHANNELS))
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.setListeners(
                new ScoreIterationListener(60),
                new TimePrintingListener(240000L)
        );

        return net;
    }


    private static DataSetIterator[] prepareDataIterators() throws Exception {
        Random rand = new Random(SEED);

        File trainData = new File(TRAIN_FOLDER);
        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, rand);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        ImageRecordReader trainRR = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
        trainRR.initialize(trainSplit);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, BATCH_SIZE, 1, 2);

        VGG16ImagePreProcessor preProcessor = new VGG16ImagePreProcessor();
        trainIter.setPreProcessor(preProcessor);

        File valFolder = new File(VAL_FOLDER);
        DataSetIterator valIter = null;
        if (valFolder.exists()) {
            FileSplit valSplit = new FileSplit(valFolder, NativeImageLoader.ALLOWED_FORMATS, rand);
            ImageRecordReader valRR = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
            valRR.initialize(valSplit);

            valIter = new RecordReaderDataSetIterator(valRR, BATCH_SIZE, 1, 2);
            valIter.setPreProcessor(preProcessor);
        } else {
            System.out.println("Ingen validerings-mappe fundet.");
        }

        return new DataSetIterator[]{ trainIter, valIter };
    }


    private static void trainModelAndTrackLoss(MultiLayerNetwork net, DataSetIterator trainIter, DataSetIterator valIter) throws Exception {
        List<Double> trainLossPerEpoch = new ArrayList<>();
        List<Double> valLossPerEpoch   = new ArrayList<>();

        for (int i = 0; i < MAX_EPOCHS; i++) {
            long start = System.currentTimeMillis();
            System.out.println("=== Starting epoch " + (i + 1) + " ===");
            trainIter.reset();

            double totalLoss = 0.0;
            int batchCount = 0;

            while (trainIter.hasNext()) {
                DataSet ds = trainIter.next();
                net.fit(ds);

                totalLoss += net.score(ds);
                batchCount++;
            }

            double avgTrainLoss = totalLoss / batchCount;
            System.out.println("Epoch " + (i+1) + " - Avg Train Loss = " + avgTrainLoss);

            trainLossPerEpoch.add(avgTrainLoss);
            double avgValLoss = Double.NaN;
            if (valIter != null) {
                avgValLoss = computeAverageLoss(net, valIter, "Val");
            }
            valLossPerEpoch.add(avgValLoss);
            long end = System.currentTimeMillis();

            System.out.println("Epoch " + (i + 1) + " ended. Took " + (end - start)/1000.0 + " seconds.");
            System.out.printf("Epoch %d - Training Loss: %.4f | Val Loss: %.4f%n",
                    i+1, avgTrainLoss, avgValLoss);
        }

        plotAndSaveLossCurves(trainLossPerEpoch, valLossPerEpoch);
    }


    private static double computeAverageLoss(MultiLayerNetwork net, DataSetIterator iter, String name) {
        iter.reset();
        double totalLoss = 0.0;
        int batches = 0;
        while (iter.hasNext()) {
            DataSet ds = iter.next();
            totalLoss += net.score(ds);
            batches++;
            if (batches % 10 == 0) {
                System.out.println("Scored batch " + batches + " " + name);
            }
        }
        return (batches == 0 ? Double.NaN : totalLoss / batches);
    }


    private static void plotAndSaveLossCurves(List<Double> trainLosses, List<Double> valLosses) {
        XYSeries trainSeries = new XYSeries("Training Loss");
        XYSeries valSeries   = new XYSeries("Validation Loss");

        for (int i = 0; i < trainLosses.size(); i++) {
            trainSeries.add(i + 1, trainLosses.get(i));
            if (valLosses.size() > i) {
                valSeries.add(i + 1, valLosses.get(i));
            }
        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(trainSeries);
        if (!valLosses.isEmpty()) {
            dataset.addSeries(valSeries);
        }

        JFreeChart chart = ChartFactory.createXYLineChart(
                "Loss Curves",
                "Epoch",
                "Loss",
                dataset
        );

        try {
            File outFile = new File("lossPlot.png");
            ChartUtils.saveChartAsPNG(outFile, chart, 800, 600);
            System.out.println("Loss-curve plot gemt i fil: " + outFile.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    private static void evaluateOnValidationSet(MultiLayerNetwork net) throws Exception {
        File valFolder = new File(VAL_FOLDER);
        if (!valFolder.exists()) {
            System.out.println("Ingen validerings-mappe fundet – springer evaluateOnValidationSet over.");
            return;
        }


        FileSplit valSplit = new FileSplit(valFolder, NativeImageLoader.ALLOWED_FORMATS, new Random(SEED));
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader valRR = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
        valRR.initialize(valSplit);

        DataSetIterator valIter = new RecordReaderDataSetIterator(valRR, BATCH_SIZE, 1, 2);

        VGG16ImagePreProcessor preProcessor = new VGG16ImagePreProcessor();
        valIter.setPreProcessor(preProcessor);

        Evaluation eval = new Evaluation(2);
        while (valIter.hasNext()) {
            DataSet ds = valIter.next();
            INDArray output = net.output(ds.getFeatures());
            eval.eval(ds.getLabels(), output);
        }

        System.out.println("Resultat på valideringssæt:");
        System.out.println(eval.stats());


        System.out.println("Confusion Matrix:");
        System.out.println(eval.getConfusionMatrix());
        System.out.println("Accuracy: " + eval.accuracy());
    }

    public DogsCatsClassifier() {
        super("Dogs vs Cats Classifier");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(500, 500);

        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS));
        getContentPane().add(mainPanel);

        JPanel topPanel = new JPanel();
        topPanel.setLayout(new FlowLayout(FlowLayout.CENTER));
        buttonLoadImage = new JButton("Vælg billede...");
        topPanel.add(buttonLoadImage);
        mainPanel.add(topPanel);

        JPanel centerPanel = new JPanel();
        centerPanel.setLayout(new BorderLayout());
        imageLabel = new JLabel("", SwingConstants.CENTER);
        imageLabel.setPreferredSize(new Dimension(300, 300));
        centerPanel.add(imageLabel, BorderLayout.CENTER);
        mainPanel.add(centerPanel);

        predictionLabel = new JLabel("Ingen klassifikation endnu...", SwingConstants.CENTER);
        mainPanel.add(predictionLabel);


        fileChooser = new JFileChooser();

        buttonLoadImage.addActionListener(e -> onLoadImage());

        new DropTarget(imageLabel, new DropTargetListener() {
            @Override public void dragEnter(DropTargetDragEvent dtde) { }
            @Override public void dragOver(DropTargetDragEvent dtde) { }
            @Override public void dropActionChanged(DropTargetDragEvent dtde) { }
            @Override public void dragExit(DropTargetEvent dte) { }
            @Override public void drop(DropTargetDropEvent dtde) {
                try {
                    dtde.acceptDrop(DnDConstants.ACTION_COPY_OR_MOVE);
                    @SuppressWarnings("unchecked")
                    List<File> droppedFiles = (List<File>) dtde.getTransferable().getTransferData(DataFlavor.javaFileListFlavor);
                    if (!droppedFiles.isEmpty()) {
                        File file = droppedFiles.get(0);
                        classifyAndShow(file);
                    }
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            }
        });
    }


    private void onLoadImage() {
        int returnVal = fileChooser.showOpenDialog(this);
        if (returnVal == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            classifyAndShow(selectedFile);
        }
    }

    private void classifyAndShow(File file) {
        ImageIcon icon = new ImageIcon(file.getAbsolutePath());
        Image scaled = icon.getImage().getScaledInstance(300, 300, Image.SCALE_SMOOTH);
        imageLabel.setIcon(new ImageIcon(scaled));

        String pred = classifyImage(file);
        predictionLabel.setText("Forudsigelse: " + pred);
    }


    private String classifyImage(File file) {
        try {
            NativeImageLoader loader = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
            INDArray image = loader.asMatrix(file);

            VGG16ImagePreProcessor preProcessor = new VGG16ImagePreProcessor();
            preProcessor.transform(image);

            INDArray output = model.output(image);
            int classIndex = Nd4j.argMax(output, 1).getInt(0);

            return (classIndex == 0) ? "Kat" : "Hund";
        } catch (Exception e) {
            e.printStackTrace();
            return "Fejl ved klassifikation.";
        }
    }
}
