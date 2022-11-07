package ann

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"errors"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"log"
	"os"
	"strconv"
)

func loadCSVData(src string, learnIndex, inputLen, outputLen int) (inputs, targets *mat.Dense) {
	flen := getFieldLen(src)
	f, err := os.Open(src)
	if err != nil {
		log.Fatal(err.Error())
	}
	defer f.Close()

	rdr := csv.NewReader(f)
	rdr.FieldsPerRecord = flen

	data, err := rdr.ReadAll()
	if err != nil {
		log.Fatal(err.Error())
	}

	indata, tdata, i, j := make([]float64, inputLen*len(data)), make([]float64, outputLen*len(data)), 0, 0
	for k, x := range data {
		if k == 0 {
			continue
		}
		for n, y := range x {
			d, err := strconv.ParseFloat(y, 64)
			if err != nil {
				log.Fatal(err.Error())
			}

			if n >= learnIndex && n < flen {
				tdata[j] = d
				j++
				continue
			}
			indata[i] = d
			i++
		}
	}
	return mat.NewDense(len(data), inputLen, indata), mat.NewDense(len(data), outputLen, tdata)
}

func getFieldLen(src string) int {
	f, err := os.Open(src)
	if err != nil {
		log.Fatal(err.Error())
	}
	defer f.Close()

	frdr := bufio.NewReader(f)
	b, err := frdr.ReadBytes('\n')
	if err != nil {
		log.Fatal(err.Error())
	}

	fields := bytes.Split(b, []byte{','})
	return len(fields)
}

func SumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {
	output := &mat.Dense{}
	numRows, numCols := m.Dims()
	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			data[i] = floats.Sum(mat.Col(nil, i, m))
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			data[i] = floats.Sum(mat.Row(nil, i, m))
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}
	return output, nil
}
