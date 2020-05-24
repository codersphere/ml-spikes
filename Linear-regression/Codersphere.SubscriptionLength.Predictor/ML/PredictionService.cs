using System;
using System.IO;
using Codersphere.SubscriptionLength.Predictor.Models;
using Microsoft.ML;

namespace Codersphere.SubscriptionLength.Predictor.ML
{
    public class PredictionService
    {
        public void Predict(SubscriptionLengthModel predictionAttributes)
        {
            var mlContext = new MLContext(1234);

            ITransformer mlModel;

            using (var stream = new FileStream("Data\\trained-model.mdl", FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                mlModel = mlContext.Model.Load(stream, out _);
            }

            var predictionEngine = mlContext
                .Model
                .CreatePredictionEngine<SubscriptionLengthModel, SubscriptionLengthPrediction>(mlModel);

            var prediction = predictionEngine.Predict(predictionAttributes);

            Console.WriteLine($"The predicted subscription length is: {prediction.SubscriptionLengthInMonths:#.##} months");
        }
    }
}
