import nj from "numjs"

// Random number between 1 and -1
// return as 2D array
export const rand = (rows: number, cols: number) => {
  const result = []
  for (let i = 0; i < rows; i++) {
    result[i] = []
    for (let ii = 0; ii < cols; ii++) {
      result[i][ii] = 2 * Math.random() - 1
    }
  }
  return result
}

// sigmoid derivative
export const curve = (numbers: nj.NdArray<number[]>) => {
  const nums = numbers.tolist()
  const result = []
  for (let i = 0; i < nums.length; i++) {
    result[i] = []
    for (let ii = 0; ii < nums[i].length; ii++) {
      result[i][ii] = nums[i][ii] * (1 - nums[i][ii])
    }
  }

  return nj.array(result)
}

// Return 2D array with 1 for positive and 0 for negative
export const binaryArray = (matrix: nj.NdArray<any>) => {
  const arr = matrix.tolist()
  const nx = arr.length
  const ny = arr[0].length

  // Loop over all cells
  for (let i = 0; i < nx; ++i) {
    for (let j = 0; j < ny; ++j) {
      if (arr[i][j] > 0) {
        arr[i][j] = 1
      } else {
        arr[i][j] = 0
      }
    }
  }

  return nj.array(arr)
}
