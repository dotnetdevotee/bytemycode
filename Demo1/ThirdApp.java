package org.example;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.*;
import ai.djl.modality.*;
import ai.djl.ndarray.*;
import ai.djl.ndarray.types.*;
import ai.djl.repository.zoo.*;
import ai.djl.translate.*;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;
import java.nio.file.Path;

public class ThirdApp {
    //

    public static void main(String[] args) throws MalformedModelException, ModelNotFoundException, IOException, TranslateException {

        Model model = Model.newInstance("PLEASE", "OnnxRuntime");
        Path x = Paths.get("/home/derek");
        model.load(x, "example");
        System.out.println("worked");

        MyTranslator mt = new MyTranslator();

        Predictor<FlowerInfo, Classifications> predictor = model.newPredictor(mt);
        FlowerInfo info = new FlowerInfo(1.0f, 2.0f, 3.0f, 4.0f);
        Object o = predictor.predict(info);

        System.out.println("done: " + o);
        
    }

    public static class MyTranslator implements Translator<FlowerInfo, Classifications> {

        private final List<String> synset;

        public MyTranslator() {
            // species name
            synset = Arrays.asList("setosa", "versicolor", "virginica");
        }

        @Override
        public NDList processInput(TranslatorContext ctx, FlowerInfo input) {
            float[] data = {input.sepalLength, input.sepalWidth, input.petalLength, input.petalWidth};
            NDArray array = ctx.getNDManager().create(data, new Shape(1, 4));
            return new NDList(array);
        }

        @Override
        public Classifications processOutput(TranslatorContext ctx, NDList list) {
            return new Classifications(synset, list.get(1));
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }
    }

    public static class FlowerInfo {

        public float sepalLength;
        public float sepalWidth;
        public float petalLength;
        public float petalWidth;

        public FlowerInfo(float sepalLength, float sepalWidth, float petalLength, float petalWidth) {
            this.sepalLength = sepalLength;
            this.sepalWidth = sepalWidth;
            this.petalLength = petalLength;
            this.petalWidth = petalWidth;
        }
    }
}