using System;
using System.IO;
using Codersphere.MovieReviews.SentimentAnalysis.Models;
using Microsoft.ML;

namespace Codersphere.MovieReviews.SentimentAnalysis.ML
{
    public class TrainingService
    {
        static readonly string trainingData = Path.Combine(Environment.CurrentDirectory, "Data", "imdb_labelled.txt");

        public ITransformer Train(MLContext mlContext)
        {
            var splitTrainTestData = LoadData(mlContext);

            var model = BuildAndTrainModel(mlContext, splitTrainTestData.TrainSet);

            Evaluate(mlContext, model, splitTrainTestData.TestSet);

            return model;
        }

        private static DataOperationsCatalog.TrainTestData LoadData(MLContext mlContext)
        {
            var dataView = mlContext.Data.LoadFromTextFile<SentimentData>(trainingData, hasHeader: false);
            var splitTrainTestData = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            return splitTrainTestData;
        }

        private static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            var estimator = mlContext.Transforms.Text.FeaturizeText(
                outputColumnName: "Features",
                inputColumnName: nameof(SentimentData.SentimentText))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                    labelColumnName: "Label", 
                    featureColumnName: "Features"));

            var model = estimator.Fit(splitTrainSet);

            return model;
        }

        private static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            var predictions = model.Transform(splitTestSet);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine();
            Console.WriteLine("=============== Model evaluation ===============");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }
    }
}
