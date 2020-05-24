using System;
using Codersphere.SubscriptionLength.Predictor.ML;
using Codersphere.SubscriptionLength.Predictor.Models;

namespace Codersphere.SubscriptionLength.Predictor
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Welcome to the subscription length predictor");

            var trainingService = new TrainingService();
            trainingService.Train();

            var predictionService = new PredictionService();
            predictionService.Predict(new SubscriptionLengthModel
            {
                IsMarried = 1,
                Age = 40,
                HasChildren = 1
            });
        }
    }
}
