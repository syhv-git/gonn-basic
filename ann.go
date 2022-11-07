package ann

func NewConfig(inputSize, hiddenSize, outputSize, epochs int, rate float64) Conf {
	return Conf{InputN: inputSize, HiddenN: hiddenSize, OutputN: outputSize, Epochs: epochs, Rate: rate}
}

func CreateNewANN(conf Conf, activator, variance func(_, _ int, v float64) float64) *ANN {
	nn := &ANN{Config: conf, Activation: activator, Variance: variance}
	return nn
}

func CreateAccurateANN(conf Conf, activator, variance func(_, _ int, v float64) float64, train, test string) (*ANN, error) {
	a := make([]*ANN, 5)
	for i := 0; i < 5; i++ {
		a[i] = CreateNewANN(conf, activator, variance)
		if err := a[i].Train(train); err != nil {
			return nil, err
		}
		a[i].Test(test)
		if a[0].Config.Accuracy >= a[i].Config.Accuracy {
			continue
		}
		a[0], a[i] = a[i], a[0]
	}
	return a[0], nil
}
