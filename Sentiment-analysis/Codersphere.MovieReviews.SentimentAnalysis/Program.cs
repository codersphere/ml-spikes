using System;
using System.Collections.Generic;
using Codersphere.MovieReviews.SentimentAnalysis.ML;
using Codersphere.MovieReviews.SentimentAnalysis.Models;
using Microsoft.ML;

namespace Codersphere.MovieReviews.SentimentAnalysis
{
    class Program
    {

        static void Main(string[] args)
        {
            Console.WriteLine("Welcome to the Codersphere movie reviews sentiment analysis spike");

            var mlContext = new MLContext();

            var trainingService = new TrainingService();
            var model = trainingService.Train(mlContext);

            var predictionService = new PredictionService();

            predictionService.Predict(
                mlContext, 
                model,
                new SentimentData
                {
                    SentimentText = "It was a brilliant movie, story was fantastic"
                });
        }
    }
}
