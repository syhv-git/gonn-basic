package test

import (
	ann "github.com/syhv-git/gonn-basic"
	"math"
	"testing"
)

func TestANN(t *testing.T) {
	train, test := "data/train/training_data.csv", "data/test/test_data.csv"
	conf := ann.NewConfig(4, 3, 3, 5000, 0.3)
	nn := ann.CreateNewANN(conf, func(_, _ int, v float64) float64 {
		return math.Max(0, v)
	}, func(_, _ int, v float64) float64 {
		if v > 0 {
			return v
		}
		return 0
	})
	if err := nn.Train(train); err != nil {
		t.Fatal(err.Error())
	}
	nn.Test(test)
	if nn.Config.Accuracy == 0 {
		t.Error("Neural Network test failed")
	}
}

func TestAccurateANN(t *testing.T) {
	train, test := "data/train/training_data.csv", "data/test/test_data.csv"
	conf := ann.NewConfig(4, 3, 3, 5000, 0.3)
	nn, err := ann.CreateAccurateANN(conf, func(_, _ int, v float64) float64 {
		return math.Max(0, v)
	}, func(_, _ int, v float64) float64 {
		if v > 0 {
			return v
		}
		return 0
	}, train, test)
	if err != nil {
		t.Fatal(err.Error())
	}
	if nn.Config.Accuracy == 0 {
		t.Error("Neural Network test failed")
	}
}
