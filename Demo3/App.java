package org.example;

import com.google.cloud.automl.v1.AnnotationPayload;
import com.google.cloud.automl.v1.ExamplePayload;
import com.google.cloud.automl.v1.ModelName;
import com.google.cloud.automl.v1.PredictRequest;
import com.google.cloud.automl.v1.PredictResponse;
import com.google.cloud.automl.v1.PredictionServiceClient;
import com.google.cloud.automl.v1.TextSnippet;
import java.io.IOException;

public class App 
{
    public static void main( String[] args ) throws IOException {
        
        String projectId = "first-automl-294917";
        String modelId = "TCN6147116623221227520";
        //String content = "Solar radiation barely reaches the surface of Uranus.";
        String content = "Uranus is where the sun doesn't shine.";
        predict(projectId, modelId, content);
    }

    static void predict(String projectId, String modelId, String content) throws IOException {
        // Initialize client that will be used to send requests. This client only needs to be created
        // once, and can be reused for multiple requests. After completing all of your requests, call
        // the "close" method on the client to safely clean up any remaining background resources.
        try (PredictionServiceClient client = PredictionServiceClient.create()) {
            // Get the full path of the model.
            ModelName name = ModelName.of(projectId, "us-central1", modelId);

            // For available mime types, see:
            // https://cloud.google.com/automl/docs/reference/rest/v1/projects.locations.models/predict#textsnippet
            TextSnippet textSnippet =
                    TextSnippet.newBuilder()
                            .setContent(content)
                            .setMimeType("text/plain") // Types: text/plain, text/html
                            .build();
            ExamplePayload payload = ExamplePayload.newBuilder().setTextSnippet(textSnippet).build();
            PredictRequest predictRequest =
                    PredictRequest.newBuilder().setName(name.toString()).setPayload(payload).build();

            PredictResponse response = client.predict(predictRequest);

            for (AnnotationPayload annotationPayload : response.getPayloadList()) {
                System.out.format("Predicted class name: %s\n", annotationPayload.getDisplayName());
                System.out.format(
                        "Predicted sentiment score: %.2f\n\n",
                        annotationPayload.getClassification().getScore());
            }
        }
    }
}
