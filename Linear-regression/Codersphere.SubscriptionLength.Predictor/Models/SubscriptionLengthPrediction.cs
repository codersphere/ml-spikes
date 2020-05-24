using Microsoft.ML.Data;

namespace Codersphere.SubscriptionLength.Predictor.Models
{
    public class SubscriptionLengthPrediction
    {
        [ColumnName("Score")]
        public float SubscriptionLengthInMonths;
    }
}