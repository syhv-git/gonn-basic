package ann

import (
	"encoding/json"
	"errors"
	"github.com/google/uuid"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"log"
	"math/rand"
	"os"
	"path"
	"time"
)

type Conf struct {
	Accuracy float64
	InputN   int
	HiddenN  int
	OutputN  int
	Epochs   int
	Rate     float64
}

type ANN struct {
	Config     Conf
	WHidden    *mat.Dense
	BHidden    *mat.Dense
	WOut       *mat.Dense
	BOut       *mat.Dense
	DBID       uuid.UUID
	Activation func(_, _ int, v float64) float64
	Variance   func(_, _ int, v float64) float64
}

func (nn *ANN) Train(src string) error {
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)
	wHidden, bHidden := mat.NewDense(nn.Config.InputN, nn.Config.HiddenN, nil), mat.NewDense(1, nn.Config.HiddenN, nil)
	wOut, bOut := mat.NewDense(nn.Config.HiddenN, nn.Config.OutputN, nil), mat.NewDense(1, nn.Config.OutputN, nil)
	rawData := [][]float64{
		wHidden.RawMatrix().Data,
		bHidden.RawMatrix().Data,
		wOut.RawMatrix().Data,
		bOut.RawMatrix().Data,
	}
	for _, d := range rawData {
		for i := range d {
			d[i] = randGen.Float64() //824634520832
		}
	}

	output := &mat.Dense{}
	x, y := loadCSVData(src, nn.Config.InputN, nn.Config.InputN, nn.Config.OutputN)
	if err := nn.backpropagate(x, y, wHidden, bHidden, wOut, bOut, output); err != nil {
		return err
	}
	return nil
}

func (nn *ANN) backpropagate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {
	var err error
	addbh := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
	addbo := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }

	for i := 0; i < nn.Config.Epochs; i++ {
		mat1, mat2, mat3, mat4, matErr := &mat.Dense{}, &mat.Dense{}, &mat.Dense{}, &mat.Dense{}, &mat.Dense{}
		mat1.Mul(x, wHidden)
		mat1.Apply(addbh, mat1)
		mat2.Apply(nn.Activation, mat1)
		mat1 = &mat.Dense{}

		mat1.Mul(mat2, wOut)
		mat1.Apply(addbo, mat1)
		output.Apply(nn.Activation, mat1)
		mat1 = &mat.Dense{}

		matErr.Sub(y, output)
		mat1.Apply(nn.Variance, output)
		mat3.Apply(nn.Variance, mat2)

		mat4.MulElem(matErr, mat1)
		matErr = &mat.Dense{}
		matErr.Mul(mat4, wOut.T())
		mat1 = &mat.Dense{}

		mat1.MulElem(matErr, mat3)
		mat3 = &mat.Dense{}
		mat3.Mul(mat2.T(), mat4)
		mat2 = &mat.Dense{}
		mat3.Scale(nn.Config.Rate, mat3)
		wOut.Add(wOut, mat3)
		mat3 = &mat.Dense{}

		mat3, err = SumAlongAxis(0, mat4)
		if err != nil {
			return err
		}
		mat3.Scale(nn.Config.Rate, mat3)
		bOut.Add(bOut, mat3)

		mat2.Mul(x.T(), mat1)
		mat2.Scale(nn.Config.Rate, mat2)
		wHidden.Add(wHidden, mat2)

		mat2 = &mat.Dense{}
		mat2, err = SumAlongAxis(0, mat1)
		if err != nil {
			return err
		}
		mat2.Scale(nn.Config.Rate, mat2)
		bHidden.Add(bHidden, mat2)
	}

	nn.WHidden = wHidden
	nn.BHidden = bHidden
	nn.WOut = wOut
	nn.BOut = bOut
	return nil
}

func (nn *ANN) Test(src string) {
	var truePred int
	predictions, err := nn.Predict(src)
	if err != nil {
		log.Fatal(err.Error())
	}

	_, testL := loadCSVData(src, nn.Config.InputN, nn.Config.InputN, nn.Config.OutputN)
	numPred, _ := predictions.Dims()
	for i := 0; i < numPred; i++ {
		var species int
		rlabel := mat.Row(nil, i, testL)
		for idx, label := range rlabel {
			if label == 1.0 {
				species = idx
				break
			}
		}
		if predictions.At(i, species) == floats.Max(mat.Row(nil, i, predictions)) {
			truePred++
		}
	}
	nn.Config.Accuracy = float64(truePred) / float64(numPred)
}

func (nn *ANN) Predict(src string) (*mat.Dense, error) {
	mat1, mat2, output := &mat.Dense{}, &mat.Dense{}, &mat.Dense{}
	addBHidden := func(_, col int, v float64) float64 { return v + nn.BHidden.At(0, col) }
	addBOut := func(_, col int, v float64) float64 { return v + nn.BOut.At(0, col) }
	if nn.WHidden == nil || nn.WOut == nil || nn.BHidden == nil || nn.BOut == nil {
		return nil, errors.New("the neural network is untrained")
	}

	x, _ := loadCSVData(src, nn.Config.InputN, nn.Config.InputN, nn.Config.OutputN)
	mat1.Mul(x, nn.WHidden)
	mat1.Apply(addBHidden, mat1)
	mat2.Apply(nn.Activation, mat1)

	mat1.Mul(mat2, nn.WOut)
	mat1.Apply(addBOut, mat1)
	output.Apply(nn.Activation, mat1)
	return output, nil
}

// Store saves the ANN as a JSON file or to a database.
// dst contains either the file path or uuid.UUID in string form
// uuid defines if dst is an ID.
// creator defines the function to create or update.
//
// The User of this function must maintain whether the record should be created or updated.
func (nn *ANN) Store(dst string, id bool, creator func(dto *DTOANN) error) error {
	dto := &DTOANN{Conf: Conf{}}
	if err := dto.load(nn); err != nil {
		return err
	}
	if id {
		dto.setConf(nn.Config)
		if dst != "" {
			uid, err := uuid.Parse(dst)
			if err != nil {
				return err
			}
			dto.ID = uid
		}
		if err := creator(dto); err != nil {
			return err
		}
		nn.DBID = dto.ID
		return nil
	}
	if path.Ext(dst) != ".json" {
		return errors.New("filename does not have the .json extension")
	}
	dst = "network/shared/saved/" + dst

	b, err := json.Marshal(dto)
	if err != nil {
		return err
	}
	if err = os.WriteFile(dst, b, 0666); err != nil {
		return err
	}
	return nil
}

// Load reads saved ANN from file or database.
// src contains either the file path or uuid.UUID in string form
// id defines if src is an uuid
// loader is the function to read from a database
func (nn *ANN) Load(src string, id bool, loader func(dto *DTOANN) error) (err error) {
	dto := &DTOANN{Conf: Conf{}}
	if id {
		dto.ID, err = uuid.Parse(src)
		if err != nil {
			return err
		}
		if err = loader(dto); err != nil {
			return err
		}

		if err = dto.unload(nn); err != nil {
			return err
		}
		return nil
	}

	if path.Ext(src) != ".json" {
		return errors.New("filename must be have the .json extension")
	}
	src = "network/shared/saved/" + src
	b, err := os.ReadFile(src)
	if err != nil {
		return err
	}

	if err = json.Unmarshal(b, dto); err != nil {
		return err
	}
	if err = dto.unload(nn); err != nil {
		return err
	}
	return nil
}

// Update provides a databasing interface
func (nn *ANN) Update(updater func(dto *DTOANN) error) error {
	dto := &DTOANN{Conf: Conf{}}
	if err := dto.load(nn); err != nil {
		return err
	}
	return updater(dto)
}

// Delete provides a databasing interface
func (nn *ANN) Delete(deleter func(dto *DTOANN) error) error {
	dto := &DTOANN{Conf: Conf{}}
	if err := dto.load(nn); err != nil {
		return err
	}
	return deleter(dto)
}

func (nn *ANN) setConf(c Conf) {
	nn.Config.Accuracy, nn.Config.InputN, nn.Config.HiddenN, nn.Config.OutputN, nn.Config.Epochs, nn.Config.Rate = c.Accuracy, c.InputN, c.HiddenN, c.OutputN, c.Epochs, c.Rate
}

// DTOANN is a Data-Transfer Object for database storage
type DTOANN struct {
	ID        uuid.UUID
	CreatedAt string
	Conf
	WHidden []byte
	BHidden []byte
	WOut    []byte
	BOut    []byte
}

// BeforeCreate is a gorm hook
func (dto *DTOANN) BeforeCreate() {
	dto.ID = uuid.New()
	dto.CreatedAt = time.Now().Format(time.RFC3339)
}

func (dto *DTOANN) load(nn *ANN) (err error) {
	dto.ID = nn.DBID
	dto.setConf(nn.Config)
	dto.WHidden, err = nn.WHidden.MarshalBinary()
	if err != nil {
		return
	}
	dto.BHidden, err = nn.BHidden.MarshalBinary()
	if err != nil {
		return
	}
	dto.WOut, err = nn.WOut.MarshalBinary()
	if err != nil {
		return
	}
	dto.BOut, err = nn.BOut.MarshalBinary()
	if err != nil {
		return
	}
	return
}

func (dto *DTOANN) unload(nn *ANN) (err error) {
	nn.DBID = dto.ID
	nn.setConf(dto.Conf)
	nn.WHidden = &mat.Dense{}
	err = nn.WHidden.UnmarshalBinary(dto.WHidden)
	if err != nil {
		return
	}
	nn.BHidden = &mat.Dense{}
	err = nn.BHidden.UnmarshalBinary(dto.BHidden)
	if err != nil {
		return
	}
	nn.WOut = &mat.Dense{}
	err = nn.WOut.UnmarshalBinary(dto.WOut)
	if err != nil {
		return
	}
	nn.BOut = &mat.Dense{}
	err = nn.BOut.UnmarshalBinary(dto.BOut)
	if err != nil {
		return
	}
	return
}

func (dto *DTOANN) setConf(c Conf) {
	dto.Accuracy, dto.InputN, dto.HiddenN, dto.OutputN, dto.Epochs, dto.Rate = c.Accuracy, c.InputN, c.HiddenN, c.OutputN, c.Epochs, c.Rate
}
