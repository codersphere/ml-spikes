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
            var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            var prediction = predictionEngine.Predict(textToTest);
            DisplayPrediction(prediction);
        }

        private void DisplayPrediction(SentimentPrediction prediction)
        {
            Console.WriteLine($"Text: {prediction.SentimentText}");
            Console.WriteLine($"Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")}");
            Console.WriteLine($"Probability: {DisplayProbability(prediction)}");
            Console.WriteLine("----------------------------------------------");
        }

        private string DisplayProbability(SentimentPrediction resultPrediction)
        {
            if (resultPrediction.Prediction)
            {
                return $"{(resultPrediction.Probability * 100):#.##}%";
            }

            return $"{((1-resultPrediction.Probability) * 100):#.##}%";
        }
    }
}
