using Microsoft.ML.Data;

namespace Codersphere.SubscriptionLength.Predictor.Models
{
    public class SubscriptionLengthModel
    {
        [LoadColumn(0)]
        public float SubscriptionLengthInMonths { get; set; }

        [LoadColumn(1)]
        public float Age { get; set; }

        [LoadColumn(2)]
        public float IsMarried { get; set; }

        [LoadColumn(3)]
        public float HasChildren { get; set; }
    }
}
