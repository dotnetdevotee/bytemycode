package org.example;

import ai.djl.*;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.nn.*;
import ai.djl.nn.core.*;

import ai.djl.basicdataset.*;
import ai.djl.ndarray.types.*;
import ai.djl.training.*;
import ai.djl.training.dataset.*;
import ai.djl.training.loss.*;
import ai.djl.training.listener.*;
import ai.djl.training.evaluator.*;
import ai.djl.training.util.*;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.nio.file.*;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class App 
{
    public static void main( String[] args ) throws IOException, TranslateException, MalformedModelException {
        long inputSize = 28*28;
        long outputSize = 10;

        SequentialBlock block = new SequentialBlock();

        block.add(Blocks.batchFlattenBlock(inputSize));
        block.add(Linear.builder().setUnits(128).build());
        block.add(Activation::relu);
        block.add(Linear.builder().setUnits(64).build());
        block.add(Activation::relu);
        block.add(Linear.builder().setUnits(outputSize).build());

        int batchSize = 32;
        Mnist mnist = Mnist.builder().setSampling(batchSize, true).build();
        mnist.prepare(new ProgressBar());

        Model model = Model.newInstance("mlp");
        model.setBlock(block);

        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                //softmaxCrossEntropyLoss is a standard loss for classification problems
                .addEvaluator(new Accuracy()) // Use accuracy so we humans can understand how accurate the model is
                .addTrainingListeners(TrainingListener.Defaults.logging());

        // Now that we have our training configuration, we should create a new trainer for our model
        Trainer trainer = model.newTrainer(config);

        trainer.initialize(new Shape(1, 28 * 28));

        int epoch = 2;

        for (int i = 0; i < epoch; ++i) {
            int index = 0;

            // We iterate through the dataset once during each epoch
            for (Batch batch : trainer.iterateDataset(mnist)) {

                // During trainBatch, we update the loss and evaluators with the results for the training batch.
                EasyTrain.trainBatch(trainer, batch);

                // Now, we update the model parameters based on the results of the latest trainBatch
                trainer.step();

                // We must make sure to close the batch to ensure all the memory associated with the batch is cleared quickly.
                // If the memory isn't closed after each batch, you will very quickly run out of memory on your GPU
                batch.close();
            }

            // Call the end epoch event for the training listeners now that we are done
            trainer.notifyListeners(listener -> listener.onEpoch(trainer));
        }

        Path modelDir = Paths.get("build/mlp");
        Files.createDirectories(modelDir);

        model.setProperty("Epoch", String.valueOf(epoch));

        model.save(modelDir, "mlp");

        var img = ImageFactory.getInstance().fromUrl("https://djl-ai.s3.amazonaws.com/resources/images/0.png");
        img.getWrappedImage();

        Path modelDir2 = Paths.get("build/mlp");
        Model model2 = Model.newInstance("mlp");
        model2.setBlock(block);
        model2.load(modelDir);

        Translator<Image, Classifications> translator = new Translator<Image, Classifications>() {

            @Override
            public NDList processInput(TranslatorContext ctx, Image input) {
                // Convert Image to NDArray
                NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.GRAYSCALE);
                return new NDList(NDImageUtils.toTensor(array));
            }

            @Override
            public Classifications processOutput(TranslatorContext ctx, NDList list) {
                // Create a Classifications with the output probabilities
                NDArray probabilities = list.singletonOrThrow().softmax(0);
                List<String> classNames = IntStream.range(0, 10).mapToObj(String::valueOf).collect(Collectors.toList());
                return new Classifications(classNames, probabilities);
            }

            @Override
            public Batchifier getBatchifier() {
                // The Batchifier describes how to combine a batch together
                // Stacking, the most common batchifier, takes N [X1, X2, ...] arrays to a single [N, X1, X2, ...] array
                return Batchifier.STACK;
            }
        };

        var predictor = model.newPredictor(translator);

        var classifications = predictor.predict(img);

        System.out.println("zero: " + classifications);

    }
}