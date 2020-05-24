using System;
using Codersphere.SubscriptionLength.Predictor.Models;
using Microsoft.ML;

namespace Codersphere.SubscriptionLength.Predictor.ML
{
    public class TrainingService
    {
        public void Train()
        {
            var mlContext = new MLContext(1234);

            var trainingData = mlContext.Data.LoadFromTextFile<SubscriptionLengthModel>(
                "Data\\subscription-length-dataset.csv",
                ',',
                hasHeader: true);

            var trainAndTestDataSplit = mlContext.Data.TrainTestSplit(trainingData, testFraction: 0.4);

            var dataProcessPipeline = mlContext.Transforms
                .CopyColumns("Label", nameof(SubscriptionLengthModel.SubscriptionLengthInMonths))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(SubscriptionLengthModel.Age)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(SubscriptionLengthModel.IsMarried)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(SubscriptionLengthModel.HasChildren)))
                .Append(mlContext.Transforms.Concatenate("Features",
                nameof(SubscriptionLengthModel.Age),
                    nameof(SubscriptionLengthModel.IsMarried),
                    nameof(SubscriptionLengthModel.HasChildren)));

            var trainer = mlContext.Regression.Trainers.Sdca(
                labelColumnName: "Label",
                featureColumnName: "Features");

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            ITransformer trainedModel = trainingPipeline.Fit(trainAndTestDataSplit.TrainSet);

            mlContext.Model.Save(
                trainedModel, 
                trainAndTestDataSplit.TrainSet.Schema, 
                "Data\\trained-model.mdl");

            var testSetTransform = trainedModel.Transform(trainAndTestDataSplit.TestSet);

            var metrics = mlContext.Regression.Evaluate(testSetTransform);

            Console.WriteLine($"Mean squared error (MSE): {metrics.MeanSquaredError:#.##}");
            Console.WriteLine($"Mean absolute error (MSE): {metrics.MeanAbsoluteError:#.##}");
            Console.WriteLine($"Root mean squared error (RMSE): {metrics.RootMeanSquaredError:#.##}");
        }
    }
}
