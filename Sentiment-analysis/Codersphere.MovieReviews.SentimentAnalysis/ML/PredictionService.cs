using System;
using System.Collections.Generic;
using Codersphere.MovieReviews.SentimentAnalysis.Models;
using Microsoft.ML;

namespace Codersphere.MovieReviews.SentimentAnalysis.ML
{
    public class PredictionService
    {
        public void Predict(MLContext mlContext, ITransformer model, SentimentData textToTest)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            var prediction = predictionFunction.Predict(textToTest);
            DisplayPrediction(prediction);
        }

        public void Predict(MLContext mlContext, ITransformer model, IEnumerable<SentimentData> textItemsToTest)
        {

            var batchComments = mlContext.Data.LoadFromEnumerable(textItemsToTest);
            var predictions = model.Transform(batchComments);

            var predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);


            foreach (var prediction in predictedResults)
            {
                DisplayPrediction(prediction);
            }
        }

        private string DisplayProbability(SentimentPrediction resultPrediction)
        {
            if (resultPrediction.Prediction)
            {
                return $"{(resultPrediction.Probability * 100):#.##}%";
            }

            return $"{((1-resultPrediction.Probability) * 100):#.##}%";
        }

        private void DisplayPrediction(SentimentPrediction prediction)
        {
            Console.WriteLine($"Text: {prediction.SentimentText}");
            Console.WriteLine($"Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")}");
            Console.WriteLine($"Probability: {DisplayProbability(prediction)}");
            Console.WriteLine("----------------------------------------------");
        }
    }
}
